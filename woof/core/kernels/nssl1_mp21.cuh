// Generated closed public-WRF v4.6.1 one-moment transcription.
// Source/licence: tools/nssl2_wrf461_variant_oracle/SOURCE_PIN.json.
// WRF line IDs refer to module_mp_nssl_2mom.F. Four existing repo referee fixes
// are applied identically; see tools/nssl2_wrf461_column_oracle/wrf_bugfix.py.
namespace nssl1 {
__device__ const float init_ax[28]={0x0.0p+0f,0x1.9bf3d80000000p+10f,0x0.0p+0f,0x0.0p+0f,0x1.3b6b860000000p+7f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_bb[28]={0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5f212e0000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x1.5555560000000p-2f};
__device__ const float init_bx[28]={0x0.0p+0f,0x1.b333340000000p-1f,0x0.0p+0f,0x0.0p+0f,0x1.3333340000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_cdx[28]={0x0.0p+0f,0x1.3333340000000p-1f,0x0.0p+0f,0x1.0000000000000p+1f,0x1.99999a0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_cno[28]={0x1.7d78400000000p+26f,0x1.86a0000000000p+19f,0x1.7d78400000000p+26f,0x1.6e36000000000p+21f,0x1.86a0000000000p+18f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_da0[28]={0x1.ce35c00000000p-1f,0x1.361f3a0000000p-1f,0x1.cfe3ac0000000p-1f,0x1.361f3a0000000p-1f,0x1.361f3a0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_da1[28]={0x1.812d080000000p+0f,0x1.83a70c0000000p+2f,0x1.8703720000000p+0f,0x1.83a70c0000000p+2f,0x1.83a70c0000000p+2f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_dab0[784]={0x0.0p+0f,0x1.f739de0000000p-1f,0x1.97ca300000000p+0f,0x1.f739de0000000p-1f,0x1.f739de0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739de0000000p-1f,0x0.0p+0f,0x1.f69d8c0000000p-1f,0x1.361f3c0000000p-1f,0x1.361f3c0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.97ca300000000p+0f,0x1.f69d8c0000000p-1f,0x0.0p+0f,0x1.f69d8c0000000p-1f,0x1.f69d8c0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739de0000000p-1f,0x1.361f3c0000000p-1f,0x1.f69d8c0000000p-1f,0x0.0p+0f,0x1.361f3c0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739de0000000p-1f,0x1.361f3c0000000p-1f,0x1.f69d8c0000000p-1f,0x1.361f3c0000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_dab1[784]={0x0.0p+0f,0x1.4f7bbc0000000p+0f,0x1.0fdbfa0000000p+1f,0x1.4f7bbc0000000p+0f,0x1.4f7bbc0000000p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739e40000000p+1f,0x0.0p+0f,0x1.f69d900000000p+1f,0x1.361f3e0000000p+1f,0x1.361f3e0000000p+1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.11cf580000000p+1f,0x1.517b020000000p+0f,0x0.0p+0f,0x1.517b020000000p+0f,0x1.517b020000000p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739e40000000p+1f,0x1.361f3e0000000p+1f,0x1.f69d900000000p+1f,0x0.0p+0f,0x1.361f3e0000000p+1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x1.f739e40000000p+1f,0x1.361f3e0000000p+1f,0x1.f69d900000000p+1f,0x1.361f3e0000000p+1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_dnu[28]={0x1.0000000000000p+1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_fx[28]={0x0.0p+0f,0x1.0ef43a0000000p+7f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const int init_ido[28]={1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const int init_ipc[28]={2,3,1,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const int init_lliq[26]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const int init_ln[28]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const int init_lvol[28]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const int init_lz[28]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
__device__ const float init_mmgraupvt[27]={0x1.9000000000000p+5f,0x1.2c00000000000p+7f,0x1.f400000000000p+7f,0x1.5e00000000000p+8f,0x1.c200000000000p+8f,0x1.1300000000000p+9f,0x1.4500000000000p+9f,0x1.7700000000000p+9f,0x1.a900000000000p+9f,0x1.f7624e0000000p+5f,0x1.787cee0000000p+6f,0x1.caf5c20000000p+6f,0x1.066b860000000p+7f,0x1.22851e0000000p+7f,0x1.3b6b860000000p+7f,0x1.51f5c20000000p+7f,0x1.66b8520000000p+7f,0x1.7a0a3e0000000p+7f,0x1.5b3bb80000000p-1f,0x1.4699840000000p-1f,0x1.3e72da0000000p-1f,0x1.398c7e0000000p-1f,0x1.3620ee0000000p-1f,0x1.3389b60000000p-1f,0x1.31797c0000000p-1f,0x1.2fc5040000000p-1f,0x1.2e53660000000p-1f};
__device__ const float init_qxmin[28]={0x1.12e0be0000000p-30f,0x1.ad7f2a0000000p-24f,0x1.1979980000000p-40f,0x1.ad7f2a0000000p-24f,0x1.ad7f2a0000000p-24f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f,0x1.1979980000000p-40f};
__device__ const float init_xdn0[28]={0x1.f400000000000p+9f,0x1.f400000000000p+9f,0x1.c200000000000p+9f,0x1.9000000000000p+6f,0x1.f400000000000p+8f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f};
__device__ const float init_xdnmn[28]={0x1.f400000000000p+9f,0x1.f400000000000p+9f,0x1.9000000000000p+6f,0x1.9000000000000p+6f,0x1.5400000000000p+7f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f};
__device__ const float init_xdnmx[28]={0x1.f400000000000p+9f,0x1.f400000000000p+9f,0x1.ca80000000000p+9f,0x1.2c00000000000p+8f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f,0x1.c200000000000p+9f};
__device__ const float init_xmu[28]={0x1.0000000000000p+0f,0x1.5555560000000p-2f,0x1.0000000000000p+0f,0x1.5555560000000p-2f,0x1.5555560000000p-2f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_xnu[28]={0x0.0p+0f,-0x1.5555560000000p-1f,0x0.0p+0f,-0x1.5555560000000p-1f,-0x1.5555560000000p-1f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_xvmn[28]={0x1.35140c0000000p-55f,0x1.2dd5920000000p-42f,0x1.2dd5920000000p-51f,0x1.2dd5920000000p-51f,0x1.f1685a0000000p-37f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
__device__ const float init_xvmx[28]={0x1.fd58680000000p-41f,0x1.e5bfe60000000p-24f,0x1.1fda060000000p-28f,0x1.191ae40000000p-21f,0x1.191ae40000000p-18f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f};
template<int NZ> struct Column { int global_k=1, global_nz=1; float below_t=0, below2_t=0, below_w=0, above_w=0, below_in=0, above_in=0, below_qc=0;
struct coeff_ax { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x0.0p+0f;case 1:return 0x1.9bf3d80000000p+10f;case 2:return 0x0.0p+0f;case 3:return 0x0.0p+0f;case 4:return 0x1.3b6b860000000p+7f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_bb { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.5555560000000p-2f;case 1:return 0x1.5555560000000p-2f;case 2:return 0x1.5f212e0000000p-2f;case 3:return 0x1.5555560000000p-2f;case 4:return 0x1.5555560000000p-2f;case 5:return 0x1.5555560000000p-2f;case 6:return 0x1.5555560000000p-2f;case 7:return 0x1.5555560000000p-2f;case 8:return 0x1.5555560000000p-2f;case 9:return 0x1.5555560000000p-2f;case 10:return 0x1.5555560000000p-2f;case 11:return 0x1.5555560000000p-2f;case 12:return 0x1.5555560000000p-2f;case 13:return 0x1.5555560000000p-2f;case 14:return 0x1.5555560000000p-2f;case 15:return 0x1.5555560000000p-2f;case 16:return 0x1.5555560000000p-2f;case 17:return 0x1.5555560000000p-2f;case 18:return 0x1.5555560000000p-2f;case 19:return 0x1.5555560000000p-2f;case 20:return 0x1.5555560000000p-2f;case 21:return 0x1.5555560000000p-2f;case 22:return 0x1.5555560000000p-2f;case 23:return 0x1.5555560000000p-2f;case 24:return 0x1.5555560000000p-2f;case 25:return 0x1.5555560000000p-2f;case 26:return 0x1.5555560000000p-2f;case 27:return 0x1.5555560000000p-2f;default: __builtin_unreachable();}}};
struct coeff_bx { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x0.0p+0f;case 1:return 0x1.b333340000000p-1f;case 2:return 0x0.0p+0f;case 3:return 0x0.0p+0f;case 4:return 0x1.3333340000000p-1f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_cdx { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x0.0p+0f;case 1:return 0x1.3333340000000p-1f;case 2:return 0x0.0p+0f;case 3:return 0x1.0000000000000p+1f;case 4:return 0x1.99999a0000000p-1f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_cno { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.7d78400000000p+26f;case 1:return 0x1.86a0000000000p+19f;case 2:return 0x1.7d78400000000p+26f;case 3:return 0x1.6e36000000000p+21f;case 4:return 0x1.86a0000000000p+18f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_da0 { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.ce35c00000000p-1f;case 1:return 0x1.361f3a0000000p-1f;case 2:return 0x1.cfe3ac0000000p-1f;case 3:return 0x1.361f3a0000000p-1f;case 4:return 0x1.361f3a0000000p-1f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_da1 { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.812d080000000p+0f;case 1:return 0x1.83a70c0000000p+2f;case 2:return 0x1.8703720000000p+0f;case 3:return 0x1.83a70c0000000p+2f;case 4:return 0x1.83a70c0000000p+2f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_dab0 { __device__ __forceinline__ float operator()(int i0,int i1) const { switch(0+(i0-(lc))*(1)+(i1-(lc))*(1*((lqmx)-(lc)+1))) {case 0:return 0x0.0p+0f;case 1:return 0x1.f739de0000000p-1f;case 2:return 0x1.97ca300000000p+0f;case 3:return 0x1.f739de0000000p-1f;case 4:return 0x1.f739de0000000p-1f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;case 28:return 0x1.f739de0000000p-1f;case 29:return 0x0.0p+0f;case 30:return 0x1.f69d8c0000000p-1f;case 31:return 0x1.361f3c0000000p-1f;case 32:return 0x1.361f3c0000000p-1f;case 33:return 0x0.0p+0f;case 34:return 0x0.0p+0f;case 35:return 0x0.0p+0f;case 36:return 0x0.0p+0f;case 37:return 0x0.0p+0f;case 38:return 0x0.0p+0f;case 39:return 0x0.0p+0f;case 40:return 0x0.0p+0f;case 41:return 0x0.0p+0f;case 42:return 0x0.0p+0f;case 43:return 0x0.0p+0f;case 44:return 0x0.0p+0f;case 45:return 0x0.0p+0f;case 46:return 0x0.0p+0f;case 47:return 0x0.0p+0f;case 48:return 0x0.0p+0f;case 49:return 0x0.0p+0f;case 50:return 0x0.0p+0f;case 51:return 0x0.0p+0f;case 52:return 0x0.0p+0f;case 53:return 0x0.0p+0f;case 54:return 0x0.0p+0f;case 55:return 0x0.0p+0f;case 56:return 0x1.97ca300000000p+0f;case 57:return 0x1.f69d8c0000000p-1f;case 58:return 0x0.0p+0f;case 59:return 0x1.f69d8c0000000p-1f;case 60:return 0x1.f69d8c0000000p-1f;case 61:return 0x0.0p+0f;case 62:return 0x0.0p+0f;case 63:return 0x0.0p+0f;case 64:return 0x0.0p+0f;case 65:return 0x0.0p+0f;case 66:return 0x0.0p+0f;case 67:return 0x0.0p+0f;case 68:return 0x0.0p+0f;case 69:return 0x0.0p+0f;case 70:return 0x0.0p+0f;case 71:return 0x0.0p+0f;case 72:return 0x0.0p+0f;case 73:return 0x0.0p+0f;case 74:return 0x0.0p+0f;case 75:return 0x0.0p+0f;case 76:return 0x0.0p+0f;case 77:return 0x0.0p+0f;case 78:return 0x0.0p+0f;case 79:return 0x0.0p+0f;case 80:return 0x0.0p+0f;case 81:return 0x0.0p+0f;case 82:return 0x0.0p+0f;case 83:return 0x0.0p+0f;case 84:return 0x1.f739de0000000p-1f;case 85:return 0x1.361f3c0000000p-1f;case 86:return 0x1.f69d8c0000000p-1f;case 87:return 0x0.0p+0f;case 88:return 0x1.361f3c0000000p-1f;case 89:return 0x0.0p+0f;case 90:return 0x0.0p+0f;case 91:return 0x0.0p+0f;case 92:return 0x0.0p+0f;case 93:return 0x0.0p+0f;case 94:return 0x0.0p+0f;case 95:return 0x0.0p+0f;case 96:return 0x0.0p+0f;case 97:return 0x0.0p+0f;case 98:return 0x0.0p+0f;case 99:return 0x0.0p+0f;case 100:return 0x0.0p+0f;case 101:return 0x0.0p+0f;case 102:return 0x0.0p+0f;case 103:return 0x0.0p+0f;case 104:return 0x0.0p+0f;case 105:return 0x0.0p+0f;case 106:return 0x0.0p+0f;case 107:return 0x0.0p+0f;case 108:return 0x0.0p+0f;case 109:return 0x0.0p+0f;case 110:return 0x0.0p+0f;case 111:return 0x0.0p+0f;case 112:return 0x1.f739de0000000p-1f;case 113:return 0x1.361f3c0000000p-1f;case 114:return 0x1.f69d8c0000000p-1f;case 115:return 0x1.361f3c0000000p-1f;case 116:return 0x0.0p+0f;case 117:return 0x0.0p+0f;case 118:return 0x0.0p+0f;case 119:return 0x0.0p+0f;case 120:return 0x0.0p+0f;case 121:return 0x0.0p+0f;case 122:return 0x0.0p+0f;case 123:return 0x0.0p+0f;case 124:return 0x0.0p+0f;case 125:return 0x0.0p+0f;case 126:return 0x0.0p+0f;case 127:return 0x0.0p+0f;case 128:return 0x0.0p+0f;case 129:return 0x0.0p+0f;case 130:return 0x0.0p+0f;case 131:return 0x0.0p+0f;case 132:return 0x0.0p+0f;case 133:return 0x0.0p+0f;case 134:return 0x0.0p+0f;case 135:return 0x0.0p+0f;case 136:return 0x0.0p+0f;case 137:return 0x0.0p+0f;case 138:return 0x0.0p+0f;case 139:return 0x0.0p+0f;case 140:return 0x0.0p+0f;case 141:return 0x0.0p+0f;case 142:return 0x0.0p+0f;case 143:return 0x0.0p+0f;case 144:return 0x0.0p+0f;case 145:return 0x0.0p+0f;case 146:return 0x0.0p+0f;case 147:return 0x0.0p+0f;case 148:return 0x0.0p+0f;case 149:return 0x0.0p+0f;case 150:return 0x0.0p+0f;case 151:return 0x0.0p+0f;case 152:return 0x0.0p+0f;case 153:return 0x0.0p+0f;case 154:return 0x0.0p+0f;case 155:return 0x0.0p+0f;case 156:return 0x0.0p+0f;case 157:return 0x0.0p+0f;case 158:return 0x0.0p+0f;case 159:return 0x0.0p+0f;case 160:return 0x0.0p+0f;case 161:return 0x0.0p+0f;case 162:return 0x0.0p+0f;case 163:return 0x0.0p+0f;case 164:return 0x0.0p+0f;case 165:return 0x0.0p+0f;case 166:return 0x0.0p+0f;case 167:return 0x0.0p+0f;case 168:return 0x0.0p+0f;case 169:return 0x0.0p+0f;case 170:return 0x0.0p+0f;case 171:return 0x0.0p+0f;case 172:return 0x0.0p+0f;case 173:return 0x0.0p+0f;case 174:return 0x0.0p+0f;case 175:return 0x0.0p+0f;case 176:return 0x0.0p+0f;case 177:return 0x0.0p+0f;case 178:return 0x0.0p+0f;case 179:return 0x0.0p+0f;case 180:return 0x0.0p+0f;case 181:return 0x0.0p+0f;case 182:return 0x0.0p+0f;case 183:return 0x0.0p+0f;case 184:return 0x0.0p+0f;case 185:return 0x0.0p+0f;case 186:return 0x0.0p+0f;case 187:return 0x0.0p+0f;case 188:return 0x0.0p+0f;case 189:return 0x0.0p+0f;case 190:return 0x0.0p+0f;case 191:return 0x0.0p+0f;case 192:return 0x0.0p+0f;case 193:return 0x0.0p+0f;case 194:return 0x0.0p+0f;case 195:return 0x0.0p+0f;case 196:return 0x0.0p+0f;case 197:return 0x0.0p+0f;case 198:return 0x0.0p+0f;case 199:return 0x0.0p+0f;case 200:return 0x0.0p+0f;case 201:return 0x0.0p+0f;case 202:return 0x0.0p+0f;case 203:return 0x0.0p+0f;case 204:return 0x0.0p+0f;case 205:return 0x0.0p+0f;case 206:return 0x0.0p+0f;case 207:return 0x0.0p+0f;case 208:return 0x0.0p+0f;case 209:return 0x0.0p+0f;case 210:return 0x0.0p+0f;case 211:return 0x0.0p+0f;case 212:return 0x0.0p+0f;case 213:return 0x0.0p+0f;case 214:return 0x0.0p+0f;case 215:return 0x0.0p+0f;case 216:return 0x0.0p+0f;case 217:return 0x0.0p+0f;case 218:return 0x0.0p+0f;case 219:return 0x0.0p+0f;case 220:return 0x0.0p+0f;case 221:return 0x0.0p+0f;case 222:return 0x0.0p+0f;case 223:return 0x0.0p+0f;case 224:return 0x0.0p+0f;case 225:return 0x0.0p+0f;case 226:return 0x0.0p+0f;case 227:return 0x0.0p+0f;case 228:return 0x0.0p+0f;case 229:return 0x0.0p+0f;case 230:return 0x0.0p+0f;case 231:return 0x0.0p+0f;case 232:return 0x0.0p+0f;case 233:return 0x0.0p+0f;case 234:return 0x0.0p+0f;case 235:return 0x0.0p+0f;case 236:return 0x0.0p+0f;case 237:return 0x0.0p+0f;case 238:return 0x0.0p+0f;case 239:return 0x0.0p+0f;case 240:return 0x0.0p+0f;case 241:return 0x0.0p+0f;case 242:return 0x0.0p+0f;case 243:return 0x0.0p+0f;case 244:return 0x0.0p+0f;case 245:return 0x0.0p+0f;case 246:return 0x0.0p+0f;case 247:return 0x0.0p+0f;case 248:return 0x0.0p+0f;case 249:return 0x0.0p+0f;case 250:return 0x0.0p+0f;case 251:return 0x0.0p+0f;case 252:return 0x0.0p+0f;case 253:return 0x0.0p+0f;case 254:return 0x0.0p+0f;case 255:return 0x0.0p+0f;case 256:return 0x0.0p+0f;case 257:return 0x0.0p+0f;case 258:return 0x0.0p+0f;case 259:return 0x0.0p+0f;case 260:return 0x0.0p+0f;case 261:return 0x0.0p+0f;case 262:return 0x0.0p+0f;case 263:return 0x0.0p+0f;case 264:return 0x0.0p+0f;case 265:return 0x0.0p+0f;case 266:return 0x0.0p+0f;case 267:return 0x0.0p+0f;case 268:return 0x0.0p+0f;case 269:return 0x0.0p+0f;case 270:return 0x0.0p+0f;case 271:return 0x0.0p+0f;case 272:return 0x0.0p+0f;case 273:return 0x0.0p+0f;case 274:return 0x0.0p+0f;case 275:return 0x0.0p+0f;case 276:return 0x0.0p+0f;case 277:return 0x0.0p+0f;case 278:return 0x0.0p+0f;case 279:return 0x0.0p+0f;case 280:return 0x0.0p+0f;case 281:return 0x0.0p+0f;case 282:return 0x0.0p+0f;case 283:return 0x0.0p+0f;case 284:return 0x0.0p+0f;case 285:return 0x0.0p+0f;case 286:return 0x0.0p+0f;case 287:return 0x0.0p+0f;case 288:return 0x0.0p+0f;case 289:return 0x0.0p+0f;case 290:return 0x0.0p+0f;case 291:return 0x0.0p+0f;case 292:return 0x0.0p+0f;case 293:return 0x0.0p+0f;case 294:return 0x0.0p+0f;case 295:return 0x0.0p+0f;case 296:return 0x0.0p+0f;case 297:return 0x0.0p+0f;case 298:return 0x0.0p+0f;case 299:return 0x0.0p+0f;case 300:return 0x0.0p+0f;case 301:return 0x0.0p+0f;case 302:return 0x0.0p+0f;case 303:return 0x0.0p+0f;case 304:return 0x0.0p+0f;case 305:return 0x0.0p+0f;case 306:return 0x0.0p+0f;case 307:return 0x0.0p+0f;case 308:return 0x0.0p+0f;case 309:return 0x0.0p+0f;case 310:return 0x0.0p+0f;case 311:return 0x0.0p+0f;case 312:return 0x0.0p+0f;case 313:return 0x0.0p+0f;case 314:return 0x0.0p+0f;case 315:return 0x0.0p+0f;case 316:return 0x0.0p+0f;case 317:return 0x0.0p+0f;case 318:return 0x0.0p+0f;case 319:return 0x0.0p+0f;case 320:return 0x0.0p+0f;case 321:return 0x0.0p+0f;case 322:return 0x0.0p+0f;case 323:return 0x0.0p+0f;case 324:return 0x0.0p+0f;case 325:return 0x0.0p+0f;case 326:return 0x0.0p+0f;case 327:return 0x0.0p+0f;case 328:return 0x0.0p+0f;case 329:return 0x0.0p+0f;case 330:return 0x0.0p+0f;case 331:return 0x0.0p+0f;case 332:return 0x0.0p+0f;case 333:return 0x0.0p+0f;case 334:return 0x0.0p+0f;case 335:return 0x0.0p+0f;case 336:return 0x0.0p+0f;case 337:return 0x0.0p+0f;case 338:return 0x0.0p+0f;case 339:return 0x0.0p+0f;case 340:return 0x0.0p+0f;case 341:return 0x0.0p+0f;case 342:return 0x0.0p+0f;case 343:return 0x0.0p+0f;case 344:return 0x0.0p+0f;case 345:return 0x0.0p+0f;case 346:return 0x0.0p+0f;case 347:return 0x0.0p+0f;case 348:return 0x0.0p+0f;case 349:return 0x0.0p+0f;case 350:return 0x0.0p+0f;case 351:return 0x0.0p+0f;case 352:return 0x0.0p+0f;case 353:return 0x0.0p+0f;case 354:return 0x0.0p+0f;case 355:return 0x0.0p+0f;case 356:return 0x0.0p+0f;case 357:return 0x0.0p+0f;case 358:return 0x0.0p+0f;case 359:return 0x0.0p+0f;case 360:return 0x0.0p+0f;case 361:return 0x0.0p+0f;case 362:return 0x0.0p+0f;case 363:return 0x0.0p+0f;case 364:return 0x0.0p+0f;case 365:return 0x0.0p+0f;case 366:return 0x0.0p+0f;case 367:return 0x0.0p+0f;case 368:return 0x0.0p+0f;case 369:return 0x0.0p+0f;case 370:return 0x0.0p+0f;case 371:return 0x0.0p+0f;case 372:return 0x0.0p+0f;case 373:return 0x0.0p+0f;case 374:return 0x0.0p+0f;case 375:return 0x0.0p+0f;case 376:return 0x0.0p+0f;case 377:return 0x0.0p+0f;case 378:return 0x0.0p+0f;case 379:return 0x0.0p+0f;case 380:return 0x0.0p+0f;case 381:return 0x0.0p+0f;case 382:return 0x0.0p+0f;case 383:return 0x0.0p+0f;case 384:return 0x0.0p+0f;case 385:return 0x0.0p+0f;case 386:return 0x0.0p+0f;case 387:return 0x0.0p+0f;case 388:return 0x0.0p+0f;case 389:return 0x0.0p+0f;case 390:return 0x0.0p+0f;case 391:return 0x0.0p+0f;case 392:return 0x0.0p+0f;case 393:return 0x0.0p+0f;case 394:return 0x0.0p+0f;case 395:return 0x0.0p+0f;case 396:return 0x0.0p+0f;case 397:return 0x0.0p+0f;case 398:return 0x0.0p+0f;case 399:return 0x0.0p+0f;case 400:return 0x0.0p+0f;case 401:return 0x0.0p+0f;case 402:return 0x0.0p+0f;case 403:return 0x0.0p+0f;case 404:return 0x0.0p+0f;case 405:return 0x0.0p+0f;case 406:return 0x0.0p+0f;case 407:return 0x0.0p+0f;case 408:return 0x0.0p+0f;case 409:return 0x0.0p+0f;case 410:return 0x0.0p+0f;case 411:return 0x0.0p+0f;case 412:return 0x0.0p+0f;case 413:return 0x0.0p+0f;case 414:return 0x0.0p+0f;case 415:return 0x0.0p+0f;case 416:return 0x0.0p+0f;case 417:return 0x0.0p+0f;case 418:return 0x0.0p+0f;case 419:return 0x0.0p+0f;case 420:return 0x0.0p+0f;case 421:return 0x0.0p+0f;case 422:return 0x0.0p+0f;case 423:return 0x0.0p+0f;case 424:return 0x0.0p+0f;case 425:return 0x0.0p+0f;case 426:return 0x0.0p+0f;case 427:return 0x0.0p+0f;case 428:return 0x0.0p+0f;case 429:return 0x0.0p+0f;case 430:return 0x0.0p+0f;case 431:return 0x0.0p+0f;case 432:return 0x0.0p+0f;case 433:return 0x0.0p+0f;case 434:return 0x0.0p+0f;case 435:return 0x0.0p+0f;case 436:return 0x0.0p+0f;case 437:return 0x0.0p+0f;case 438:return 0x0.0p+0f;case 439:return 0x0.0p+0f;case 440:return 0x0.0p+0f;case 441:return 0x0.0p+0f;case 442:return 0x0.0p+0f;case 443:return 0x0.0p+0f;case 444:return 0x0.0p+0f;case 445:return 0x0.0p+0f;case 446:return 0x0.0p+0f;case 447:return 0x0.0p+0f;case 448:return 0x0.0p+0f;case 449:return 0x0.0p+0f;case 450:return 0x0.0p+0f;case 451:return 0x0.0p+0f;case 452:return 0x0.0p+0f;case 453:return 0x0.0p+0f;case 454:return 0x0.0p+0f;case 455:return 0x0.0p+0f;case 456:return 0x0.0p+0f;case 457:return 0x0.0p+0f;case 458:return 0x0.0p+0f;case 459:return 0x0.0p+0f;case 460:return 0x0.0p+0f;case 461:return 0x0.0p+0f;case 462:return 0x0.0p+0f;case 463:return 0x0.0p+0f;case 464:return 0x0.0p+0f;case 465:return 0x0.0p+0f;case 466:return 0x0.0p+0f;case 467:return 0x0.0p+0f;case 468:return 0x0.0p+0f;case 469:return 0x0.0p+0f;case 470:return 0x0.0p+0f;case 471:return 0x0.0p+0f;case 472:return 0x0.0p+0f;case 473:return 0x0.0p+0f;case 474:return 0x0.0p+0f;case 475:return 0x0.0p+0f;case 476:return 0x0.0p+0f;case 477:return 0x0.0p+0f;case 478:return 0x0.0p+0f;case 479:return 0x0.0p+0f;case 480:return 0x0.0p+0f;case 481:return 0x0.0p+0f;case 482:return 0x0.0p+0f;case 483:return 0x0.0p+0f;case 484:return 0x0.0p+0f;case 485:return 0x0.0p+0f;case 486:return 0x0.0p+0f;case 487:return 0x0.0p+0f;case 488:return 0x0.0p+0f;case 489:return 0x0.0p+0f;case 490:return 0x0.0p+0f;case 491:return 0x0.0p+0f;case 492:return 0x0.0p+0f;case 493:return 0x0.0p+0f;case 494:return 0x0.0p+0f;case 495:return 0x0.0p+0f;case 496:return 0x0.0p+0f;case 497:return 0x0.0p+0f;case 498:return 0x0.0p+0f;case 499:return 0x0.0p+0f;case 500:return 0x0.0p+0f;case 501:return 0x0.0p+0f;case 502:return 0x0.0p+0f;case 503:return 0x0.0p+0f;case 504:return 0x0.0p+0f;case 505:return 0x0.0p+0f;case 506:return 0x0.0p+0f;case 507:return 0x0.0p+0f;case 508:return 0x0.0p+0f;case 509:return 0x0.0p+0f;case 510:return 0x0.0p+0f;case 511:return 0x0.0p+0f;case 512:return 0x0.0p+0f;case 513:return 0x0.0p+0f;case 514:return 0x0.0p+0f;case 515:return 0x0.0p+0f;case 516:return 0x0.0p+0f;case 517:return 0x0.0p+0f;case 518:return 0x0.0p+0f;case 519:return 0x0.0p+0f;case 520:return 0x0.0p+0f;case 521:return 0x0.0p+0f;case 522:return 0x0.0p+0f;case 523:return 0x0.0p+0f;case 524:return 0x0.0p+0f;case 525:return 0x0.0p+0f;case 526:return 0x0.0p+0f;case 527:return 0x0.0p+0f;case 528:return 0x0.0p+0f;case 529:return 0x0.0p+0f;case 530:return 0x0.0p+0f;case 531:return 0x0.0p+0f;case 532:return 0x0.0p+0f;case 533:return 0x0.0p+0f;case 534:return 0x0.0p+0f;case 535:return 0x0.0p+0f;case 536:return 0x0.0p+0f;case 537:return 0x0.0p+0f;case 538:return 0x0.0p+0f;case 539:return 0x0.0p+0f;case 540:return 0x0.0p+0f;case 541:return 0x0.0p+0f;case 542:return 0x0.0p+0f;case 543:return 0x0.0p+0f;case 544:return 0x0.0p+0f;case 545:return 0x0.0p+0f;case 546:return 0x0.0p+0f;case 547:return 0x0.0p+0f;case 548:return 0x0.0p+0f;case 549:return 0x0.0p+0f;case 550:return 0x0.0p+0f;case 551:return 0x0.0p+0f;case 552:return 0x0.0p+0f;case 553:return 0x0.0p+0f;case 554:return 0x0.0p+0f;case 555:return 0x0.0p+0f;case 556:return 0x0.0p+0f;case 557:return 0x0.0p+0f;case 558:return 0x0.0p+0f;case 559:return 0x0.0p+0f;case 560:return 0x0.0p+0f;case 561:return 0x0.0p+0f;case 562:return 0x0.0p+0f;case 563:return 0x0.0p+0f;case 564:return 0x0.0p+0f;case 565:return 0x0.0p+0f;case 566:return 0x0.0p+0f;case 567:return 0x0.0p+0f;case 568:return 0x0.0p+0f;case 569:return 0x0.0p+0f;case 570:return 0x0.0p+0f;case 571:return 0x0.0p+0f;case 572:return 0x0.0p+0f;case 573:return 0x0.0p+0f;case 574:return 0x0.0p+0f;case 575:return 0x0.0p+0f;case 576:return 0x0.0p+0f;case 577:return 0x0.0p+0f;case 578:return 0x0.0p+0f;case 579:return 0x0.0p+0f;case 580:return 0x0.0p+0f;case 581:return 0x0.0p+0f;case 582:return 0x0.0p+0f;case 583:return 0x0.0p+0f;case 584:return 0x0.0p+0f;case 585:return 0x0.0p+0f;case 586:return 0x0.0p+0f;case 587:return 0x0.0p+0f;case 588:return 0x0.0p+0f;case 589:return 0x0.0p+0f;case 590:return 0x0.0p+0f;case 591:return 0x0.0p+0f;case 592:return 0x0.0p+0f;case 593:return 0x0.0p+0f;case 594:return 0x0.0p+0f;case 595:return 0x0.0p+0f;case 596:return 0x0.0p+0f;case 597:return 0x0.0p+0f;case 598:return 0x0.0p+0f;case 599:return 0x0.0p+0f;case 600:return 0x0.0p+0f;case 601:return 0x0.0p+0f;case 602:return 0x0.0p+0f;case 603:return 0x0.0p+0f;case 604:return 0x0.0p+0f;case 605:return 0x0.0p+0f;case 606:return 0x0.0p+0f;case 607:return 0x0.0p+0f;case 608:return 0x0.0p+0f;case 609:return 0x0.0p+0f;case 610:return 0x0.0p+0f;case 611:return 0x0.0p+0f;case 612:return 0x0.0p+0f;case 613:return 0x0.0p+0f;case 614:return 0x0.0p+0f;case 615:return 0x0.0p+0f;case 616:return 0x0.0p+0f;case 617:return 0x0.0p+0f;case 618:return 0x0.0p+0f;case 619:return 0x0.0p+0f;case 620:return 0x0.0p+0f;case 621:return 0x0.0p+0f;case 622:return 0x0.0p+0f;case 623:return 0x0.0p+0f;case 624:return 0x0.0p+0f;case 625:return 0x0.0p+0f;case 626:return 0x0.0p+0f;case 627:return 0x0.0p+0f;case 628:return 0x0.0p+0f;case 629:return 0x0.0p+0f;case 630:return 0x0.0p+0f;case 631:return 0x0.0p+0f;case 632:return 0x0.0p+0f;case 633:return 0x0.0p+0f;case 634:return 0x0.0p+0f;case 635:return 0x0.0p+0f;case 636:return 0x0.0p+0f;case 637:return 0x0.0p+0f;case 638:return 0x0.0p+0f;case 639:return 0x0.0p+0f;case 640:return 0x0.0p+0f;case 641:return 0x0.0p+0f;case 642:return 0x0.0p+0f;case 643:return 0x0.0p+0f;case 644:return 0x0.0p+0f;case 645:return 0x0.0p+0f;case 646:return 0x0.0p+0f;case 647:return 0x0.0p+0f;case 648:return 0x0.0p+0f;case 649:return 0x0.0p+0f;case 650:return 0x0.0p+0f;case 651:return 0x0.0p+0f;case 652:return 0x0.0p+0f;case 653:return 0x0.0p+0f;case 654:return 0x0.0p+0f;case 655:return 0x0.0p+0f;case 656:return 0x0.0p+0f;case 657:return 0x0.0p+0f;case 658:return 0x0.0p+0f;case 659:return 0x0.0p+0f;case 660:return 0x0.0p+0f;case 661:return 0x0.0p+0f;case 662:return 0x0.0p+0f;case 663:return 0x0.0p+0f;case 664:return 0x0.0p+0f;case 665:return 0x0.0p+0f;case 666:return 0x0.0p+0f;case 667:return 0x0.0p+0f;case 668:return 0x0.0p+0f;case 669:return 0x0.0p+0f;case 670:return 0x0.0p+0f;case 671:return 0x0.0p+0f;case 672:return 0x0.0p+0f;case 673:return 0x0.0p+0f;case 674:return 0x0.0p+0f;case 675:return 0x0.0p+0f;case 676:return 0x0.0p+0f;case 677:return 0x0.0p+0f;case 678:return 0x0.0p+0f;case 679:return 0x0.0p+0f;case 680:return 0x0.0p+0f;case 681:return 0x0.0p+0f;case 682:return 0x0.0p+0f;case 683:return 0x0.0p+0f;case 684:return 0x0.0p+0f;case 685:return 0x0.0p+0f;case 686:return 0x0.0p+0f;case 687:return 0x0.0p+0f;case 688:return 0x0.0p+0f;case 689:return 0x0.0p+0f;case 690:return 0x0.0p+0f;case 691:return 0x0.0p+0f;case 692:return 0x0.0p+0f;case 693:return 0x0.0p+0f;case 694:return 0x0.0p+0f;case 695:return 0x0.0p+0f;case 696:return 0x0.0p+0f;case 697:return 0x0.0p+0f;case 698:return 0x0.0p+0f;case 699:return 0x0.0p+0f;case 700:return 0x0.0p+0f;case 701:return 0x0.0p+0f;case 702:return 0x0.0p+0f;case 703:return 0x0.0p+0f;case 704:return 0x0.0p+0f;case 705:return 0x0.0p+0f;case 706:return 0x0.0p+0f;case 707:return 0x0.0p+0f;case 708:return 0x0.0p+0f;case 709:return 0x0.0p+0f;case 710:return 0x0.0p+0f;case 711:return 0x0.0p+0f;case 712:return 0x0.0p+0f;case 713:return 0x0.0p+0f;case 714:return 0x0.0p+0f;case 715:return 0x0.0p+0f;case 716:return 0x0.0p+0f;case 717:return 0x0.0p+0f;case 718:return 0x0.0p+0f;case 719:return 0x0.0p+0f;case 720:return 0x0.0p+0f;case 721:return 0x0.0p+0f;case 722:return 0x0.0p+0f;case 723:return 0x0.0p+0f;case 724:return 0x0.0p+0f;case 725:return 0x0.0p+0f;case 726:return 0x0.0p+0f;case 727:return 0x0.0p+0f;case 728:return 0x0.0p+0f;case 729:return 0x0.0p+0f;case 730:return 0x0.0p+0f;case 731:return 0x0.0p+0f;case 732:return 0x0.0p+0f;case 733:return 0x0.0p+0f;case 734:return 0x0.0p+0f;case 735:return 0x0.0p+0f;case 736:return 0x0.0p+0f;case 737:return 0x0.0p+0f;case 738:return 0x0.0p+0f;case 739:return 0x0.0p+0f;case 740:return 0x0.0p+0f;case 741:return 0x0.0p+0f;case 742:return 0x0.0p+0f;case 743:return 0x0.0p+0f;case 744:return 0x0.0p+0f;case 745:return 0x0.0p+0f;case 746:return 0x0.0p+0f;case 747:return 0x0.0p+0f;case 748:return 0x0.0p+0f;case 749:return 0x0.0p+0f;case 750:return 0x0.0p+0f;case 751:return 0x0.0p+0f;case 752:return 0x0.0p+0f;case 753:return 0x0.0p+0f;case 754:return 0x0.0p+0f;case 755:return 0x0.0p+0f;case 756:return 0x0.0p+0f;case 757:return 0x0.0p+0f;case 758:return 0x0.0p+0f;case 759:return 0x0.0p+0f;case 760:return 0x0.0p+0f;case 761:return 0x0.0p+0f;case 762:return 0x0.0p+0f;case 763:return 0x0.0p+0f;case 764:return 0x0.0p+0f;case 765:return 0x0.0p+0f;case 766:return 0x0.0p+0f;case 767:return 0x0.0p+0f;case 768:return 0x0.0p+0f;case 769:return 0x0.0p+0f;case 770:return 0x0.0p+0f;case 771:return 0x0.0p+0f;case 772:return 0x0.0p+0f;case 773:return 0x0.0p+0f;case 774:return 0x0.0p+0f;case 775:return 0x0.0p+0f;case 776:return 0x0.0p+0f;case 777:return 0x0.0p+0f;case 778:return 0x0.0p+0f;case 779:return 0x0.0p+0f;case 780:return 0x0.0p+0f;case 781:return 0x0.0p+0f;case 782:return 0x0.0p+0f;case 783:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_dab1 { __device__ __forceinline__ float operator()(int i0,int i1) const { switch(0+(i0-(lc))*(1)+(i1-(lc))*(1*((lqmx)-(lc)+1))) {case 0:return 0x0.0p+0f;case 1:return 0x1.4f7bbc0000000p+0f;case 2:return 0x1.0fdbfa0000000p+1f;case 3:return 0x1.4f7bbc0000000p+0f;case 4:return 0x1.4f7bbc0000000p+0f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;case 28:return 0x1.f739e40000000p+1f;case 29:return 0x0.0p+0f;case 30:return 0x1.f69d900000000p+1f;case 31:return 0x1.361f3e0000000p+1f;case 32:return 0x1.361f3e0000000p+1f;case 33:return 0x0.0p+0f;case 34:return 0x0.0p+0f;case 35:return 0x0.0p+0f;case 36:return 0x0.0p+0f;case 37:return 0x0.0p+0f;case 38:return 0x0.0p+0f;case 39:return 0x0.0p+0f;case 40:return 0x0.0p+0f;case 41:return 0x0.0p+0f;case 42:return 0x0.0p+0f;case 43:return 0x0.0p+0f;case 44:return 0x0.0p+0f;case 45:return 0x0.0p+0f;case 46:return 0x0.0p+0f;case 47:return 0x0.0p+0f;case 48:return 0x0.0p+0f;case 49:return 0x0.0p+0f;case 50:return 0x0.0p+0f;case 51:return 0x0.0p+0f;case 52:return 0x0.0p+0f;case 53:return 0x0.0p+0f;case 54:return 0x0.0p+0f;case 55:return 0x0.0p+0f;case 56:return 0x1.11cf580000000p+1f;case 57:return 0x1.517b020000000p+0f;case 58:return 0x0.0p+0f;case 59:return 0x1.517b020000000p+0f;case 60:return 0x1.517b020000000p+0f;case 61:return 0x0.0p+0f;case 62:return 0x0.0p+0f;case 63:return 0x0.0p+0f;case 64:return 0x0.0p+0f;case 65:return 0x0.0p+0f;case 66:return 0x0.0p+0f;case 67:return 0x0.0p+0f;case 68:return 0x0.0p+0f;case 69:return 0x0.0p+0f;case 70:return 0x0.0p+0f;case 71:return 0x0.0p+0f;case 72:return 0x0.0p+0f;case 73:return 0x0.0p+0f;case 74:return 0x0.0p+0f;case 75:return 0x0.0p+0f;case 76:return 0x0.0p+0f;case 77:return 0x0.0p+0f;case 78:return 0x0.0p+0f;case 79:return 0x0.0p+0f;case 80:return 0x0.0p+0f;case 81:return 0x0.0p+0f;case 82:return 0x0.0p+0f;case 83:return 0x0.0p+0f;case 84:return 0x1.f739e40000000p+1f;case 85:return 0x1.361f3e0000000p+1f;case 86:return 0x1.f69d900000000p+1f;case 87:return 0x0.0p+0f;case 88:return 0x1.361f3e0000000p+1f;case 89:return 0x0.0p+0f;case 90:return 0x0.0p+0f;case 91:return 0x0.0p+0f;case 92:return 0x0.0p+0f;case 93:return 0x0.0p+0f;case 94:return 0x0.0p+0f;case 95:return 0x0.0p+0f;case 96:return 0x0.0p+0f;case 97:return 0x0.0p+0f;case 98:return 0x0.0p+0f;case 99:return 0x0.0p+0f;case 100:return 0x0.0p+0f;case 101:return 0x0.0p+0f;case 102:return 0x0.0p+0f;case 103:return 0x0.0p+0f;case 104:return 0x0.0p+0f;case 105:return 0x0.0p+0f;case 106:return 0x0.0p+0f;case 107:return 0x0.0p+0f;case 108:return 0x0.0p+0f;case 109:return 0x0.0p+0f;case 110:return 0x0.0p+0f;case 111:return 0x0.0p+0f;case 112:return 0x1.f739e40000000p+1f;case 113:return 0x1.361f3e0000000p+1f;case 114:return 0x1.f69d900000000p+1f;case 115:return 0x1.361f3e0000000p+1f;case 116:return 0x0.0p+0f;case 117:return 0x0.0p+0f;case 118:return 0x0.0p+0f;case 119:return 0x0.0p+0f;case 120:return 0x0.0p+0f;case 121:return 0x0.0p+0f;case 122:return 0x0.0p+0f;case 123:return 0x0.0p+0f;case 124:return 0x0.0p+0f;case 125:return 0x0.0p+0f;case 126:return 0x0.0p+0f;case 127:return 0x0.0p+0f;case 128:return 0x0.0p+0f;case 129:return 0x0.0p+0f;case 130:return 0x0.0p+0f;case 131:return 0x0.0p+0f;case 132:return 0x0.0p+0f;case 133:return 0x0.0p+0f;case 134:return 0x0.0p+0f;case 135:return 0x0.0p+0f;case 136:return 0x0.0p+0f;case 137:return 0x0.0p+0f;case 138:return 0x0.0p+0f;case 139:return 0x0.0p+0f;case 140:return 0x0.0p+0f;case 141:return 0x0.0p+0f;case 142:return 0x0.0p+0f;case 143:return 0x0.0p+0f;case 144:return 0x0.0p+0f;case 145:return 0x0.0p+0f;case 146:return 0x0.0p+0f;case 147:return 0x0.0p+0f;case 148:return 0x0.0p+0f;case 149:return 0x0.0p+0f;case 150:return 0x0.0p+0f;case 151:return 0x0.0p+0f;case 152:return 0x0.0p+0f;case 153:return 0x0.0p+0f;case 154:return 0x0.0p+0f;case 155:return 0x0.0p+0f;case 156:return 0x0.0p+0f;case 157:return 0x0.0p+0f;case 158:return 0x0.0p+0f;case 159:return 0x0.0p+0f;case 160:return 0x0.0p+0f;case 161:return 0x0.0p+0f;case 162:return 0x0.0p+0f;case 163:return 0x0.0p+0f;case 164:return 0x0.0p+0f;case 165:return 0x0.0p+0f;case 166:return 0x0.0p+0f;case 167:return 0x0.0p+0f;case 168:return 0x0.0p+0f;case 169:return 0x0.0p+0f;case 170:return 0x0.0p+0f;case 171:return 0x0.0p+0f;case 172:return 0x0.0p+0f;case 173:return 0x0.0p+0f;case 174:return 0x0.0p+0f;case 175:return 0x0.0p+0f;case 176:return 0x0.0p+0f;case 177:return 0x0.0p+0f;case 178:return 0x0.0p+0f;case 179:return 0x0.0p+0f;case 180:return 0x0.0p+0f;case 181:return 0x0.0p+0f;case 182:return 0x0.0p+0f;case 183:return 0x0.0p+0f;case 184:return 0x0.0p+0f;case 185:return 0x0.0p+0f;case 186:return 0x0.0p+0f;case 187:return 0x0.0p+0f;case 188:return 0x0.0p+0f;case 189:return 0x0.0p+0f;case 190:return 0x0.0p+0f;case 191:return 0x0.0p+0f;case 192:return 0x0.0p+0f;case 193:return 0x0.0p+0f;case 194:return 0x0.0p+0f;case 195:return 0x0.0p+0f;case 196:return 0x0.0p+0f;case 197:return 0x0.0p+0f;case 198:return 0x0.0p+0f;case 199:return 0x0.0p+0f;case 200:return 0x0.0p+0f;case 201:return 0x0.0p+0f;case 202:return 0x0.0p+0f;case 203:return 0x0.0p+0f;case 204:return 0x0.0p+0f;case 205:return 0x0.0p+0f;case 206:return 0x0.0p+0f;case 207:return 0x0.0p+0f;case 208:return 0x0.0p+0f;case 209:return 0x0.0p+0f;case 210:return 0x0.0p+0f;case 211:return 0x0.0p+0f;case 212:return 0x0.0p+0f;case 213:return 0x0.0p+0f;case 214:return 0x0.0p+0f;case 215:return 0x0.0p+0f;case 216:return 0x0.0p+0f;case 217:return 0x0.0p+0f;case 218:return 0x0.0p+0f;case 219:return 0x0.0p+0f;case 220:return 0x0.0p+0f;case 221:return 0x0.0p+0f;case 222:return 0x0.0p+0f;case 223:return 0x0.0p+0f;case 224:return 0x0.0p+0f;case 225:return 0x0.0p+0f;case 226:return 0x0.0p+0f;case 227:return 0x0.0p+0f;case 228:return 0x0.0p+0f;case 229:return 0x0.0p+0f;case 230:return 0x0.0p+0f;case 231:return 0x0.0p+0f;case 232:return 0x0.0p+0f;case 233:return 0x0.0p+0f;case 234:return 0x0.0p+0f;case 235:return 0x0.0p+0f;case 236:return 0x0.0p+0f;case 237:return 0x0.0p+0f;case 238:return 0x0.0p+0f;case 239:return 0x0.0p+0f;case 240:return 0x0.0p+0f;case 241:return 0x0.0p+0f;case 242:return 0x0.0p+0f;case 243:return 0x0.0p+0f;case 244:return 0x0.0p+0f;case 245:return 0x0.0p+0f;case 246:return 0x0.0p+0f;case 247:return 0x0.0p+0f;case 248:return 0x0.0p+0f;case 249:return 0x0.0p+0f;case 250:return 0x0.0p+0f;case 251:return 0x0.0p+0f;case 252:return 0x0.0p+0f;case 253:return 0x0.0p+0f;case 254:return 0x0.0p+0f;case 255:return 0x0.0p+0f;case 256:return 0x0.0p+0f;case 257:return 0x0.0p+0f;case 258:return 0x0.0p+0f;case 259:return 0x0.0p+0f;case 260:return 0x0.0p+0f;case 261:return 0x0.0p+0f;case 262:return 0x0.0p+0f;case 263:return 0x0.0p+0f;case 264:return 0x0.0p+0f;case 265:return 0x0.0p+0f;case 266:return 0x0.0p+0f;case 267:return 0x0.0p+0f;case 268:return 0x0.0p+0f;case 269:return 0x0.0p+0f;case 270:return 0x0.0p+0f;case 271:return 0x0.0p+0f;case 272:return 0x0.0p+0f;case 273:return 0x0.0p+0f;case 274:return 0x0.0p+0f;case 275:return 0x0.0p+0f;case 276:return 0x0.0p+0f;case 277:return 0x0.0p+0f;case 278:return 0x0.0p+0f;case 279:return 0x0.0p+0f;case 280:return 0x0.0p+0f;case 281:return 0x0.0p+0f;case 282:return 0x0.0p+0f;case 283:return 0x0.0p+0f;case 284:return 0x0.0p+0f;case 285:return 0x0.0p+0f;case 286:return 0x0.0p+0f;case 287:return 0x0.0p+0f;case 288:return 0x0.0p+0f;case 289:return 0x0.0p+0f;case 290:return 0x0.0p+0f;case 291:return 0x0.0p+0f;case 292:return 0x0.0p+0f;case 293:return 0x0.0p+0f;case 294:return 0x0.0p+0f;case 295:return 0x0.0p+0f;case 296:return 0x0.0p+0f;case 297:return 0x0.0p+0f;case 298:return 0x0.0p+0f;case 299:return 0x0.0p+0f;case 300:return 0x0.0p+0f;case 301:return 0x0.0p+0f;case 302:return 0x0.0p+0f;case 303:return 0x0.0p+0f;case 304:return 0x0.0p+0f;case 305:return 0x0.0p+0f;case 306:return 0x0.0p+0f;case 307:return 0x0.0p+0f;case 308:return 0x0.0p+0f;case 309:return 0x0.0p+0f;case 310:return 0x0.0p+0f;case 311:return 0x0.0p+0f;case 312:return 0x0.0p+0f;case 313:return 0x0.0p+0f;case 314:return 0x0.0p+0f;case 315:return 0x0.0p+0f;case 316:return 0x0.0p+0f;case 317:return 0x0.0p+0f;case 318:return 0x0.0p+0f;case 319:return 0x0.0p+0f;case 320:return 0x0.0p+0f;case 321:return 0x0.0p+0f;case 322:return 0x0.0p+0f;case 323:return 0x0.0p+0f;case 324:return 0x0.0p+0f;case 325:return 0x0.0p+0f;case 326:return 0x0.0p+0f;case 327:return 0x0.0p+0f;case 328:return 0x0.0p+0f;case 329:return 0x0.0p+0f;case 330:return 0x0.0p+0f;case 331:return 0x0.0p+0f;case 332:return 0x0.0p+0f;case 333:return 0x0.0p+0f;case 334:return 0x0.0p+0f;case 335:return 0x0.0p+0f;case 336:return 0x0.0p+0f;case 337:return 0x0.0p+0f;case 338:return 0x0.0p+0f;case 339:return 0x0.0p+0f;case 340:return 0x0.0p+0f;case 341:return 0x0.0p+0f;case 342:return 0x0.0p+0f;case 343:return 0x0.0p+0f;case 344:return 0x0.0p+0f;case 345:return 0x0.0p+0f;case 346:return 0x0.0p+0f;case 347:return 0x0.0p+0f;case 348:return 0x0.0p+0f;case 349:return 0x0.0p+0f;case 350:return 0x0.0p+0f;case 351:return 0x0.0p+0f;case 352:return 0x0.0p+0f;case 353:return 0x0.0p+0f;case 354:return 0x0.0p+0f;case 355:return 0x0.0p+0f;case 356:return 0x0.0p+0f;case 357:return 0x0.0p+0f;case 358:return 0x0.0p+0f;case 359:return 0x0.0p+0f;case 360:return 0x0.0p+0f;case 361:return 0x0.0p+0f;case 362:return 0x0.0p+0f;case 363:return 0x0.0p+0f;case 364:return 0x0.0p+0f;case 365:return 0x0.0p+0f;case 366:return 0x0.0p+0f;case 367:return 0x0.0p+0f;case 368:return 0x0.0p+0f;case 369:return 0x0.0p+0f;case 370:return 0x0.0p+0f;case 371:return 0x0.0p+0f;case 372:return 0x0.0p+0f;case 373:return 0x0.0p+0f;case 374:return 0x0.0p+0f;case 375:return 0x0.0p+0f;case 376:return 0x0.0p+0f;case 377:return 0x0.0p+0f;case 378:return 0x0.0p+0f;case 379:return 0x0.0p+0f;case 380:return 0x0.0p+0f;case 381:return 0x0.0p+0f;case 382:return 0x0.0p+0f;case 383:return 0x0.0p+0f;case 384:return 0x0.0p+0f;case 385:return 0x0.0p+0f;case 386:return 0x0.0p+0f;case 387:return 0x0.0p+0f;case 388:return 0x0.0p+0f;case 389:return 0x0.0p+0f;case 390:return 0x0.0p+0f;case 391:return 0x0.0p+0f;case 392:return 0x0.0p+0f;case 393:return 0x0.0p+0f;case 394:return 0x0.0p+0f;case 395:return 0x0.0p+0f;case 396:return 0x0.0p+0f;case 397:return 0x0.0p+0f;case 398:return 0x0.0p+0f;case 399:return 0x0.0p+0f;case 400:return 0x0.0p+0f;case 401:return 0x0.0p+0f;case 402:return 0x0.0p+0f;case 403:return 0x0.0p+0f;case 404:return 0x0.0p+0f;case 405:return 0x0.0p+0f;case 406:return 0x0.0p+0f;case 407:return 0x0.0p+0f;case 408:return 0x0.0p+0f;case 409:return 0x0.0p+0f;case 410:return 0x0.0p+0f;case 411:return 0x0.0p+0f;case 412:return 0x0.0p+0f;case 413:return 0x0.0p+0f;case 414:return 0x0.0p+0f;case 415:return 0x0.0p+0f;case 416:return 0x0.0p+0f;case 417:return 0x0.0p+0f;case 418:return 0x0.0p+0f;case 419:return 0x0.0p+0f;case 420:return 0x0.0p+0f;case 421:return 0x0.0p+0f;case 422:return 0x0.0p+0f;case 423:return 0x0.0p+0f;case 424:return 0x0.0p+0f;case 425:return 0x0.0p+0f;case 426:return 0x0.0p+0f;case 427:return 0x0.0p+0f;case 428:return 0x0.0p+0f;case 429:return 0x0.0p+0f;case 430:return 0x0.0p+0f;case 431:return 0x0.0p+0f;case 432:return 0x0.0p+0f;case 433:return 0x0.0p+0f;case 434:return 0x0.0p+0f;case 435:return 0x0.0p+0f;case 436:return 0x0.0p+0f;case 437:return 0x0.0p+0f;case 438:return 0x0.0p+0f;case 439:return 0x0.0p+0f;case 440:return 0x0.0p+0f;case 441:return 0x0.0p+0f;case 442:return 0x0.0p+0f;case 443:return 0x0.0p+0f;case 444:return 0x0.0p+0f;case 445:return 0x0.0p+0f;case 446:return 0x0.0p+0f;case 447:return 0x0.0p+0f;case 448:return 0x0.0p+0f;case 449:return 0x0.0p+0f;case 450:return 0x0.0p+0f;case 451:return 0x0.0p+0f;case 452:return 0x0.0p+0f;case 453:return 0x0.0p+0f;case 454:return 0x0.0p+0f;case 455:return 0x0.0p+0f;case 456:return 0x0.0p+0f;case 457:return 0x0.0p+0f;case 458:return 0x0.0p+0f;case 459:return 0x0.0p+0f;case 460:return 0x0.0p+0f;case 461:return 0x0.0p+0f;case 462:return 0x0.0p+0f;case 463:return 0x0.0p+0f;case 464:return 0x0.0p+0f;case 465:return 0x0.0p+0f;case 466:return 0x0.0p+0f;case 467:return 0x0.0p+0f;case 468:return 0x0.0p+0f;case 469:return 0x0.0p+0f;case 470:return 0x0.0p+0f;case 471:return 0x0.0p+0f;case 472:return 0x0.0p+0f;case 473:return 0x0.0p+0f;case 474:return 0x0.0p+0f;case 475:return 0x0.0p+0f;case 476:return 0x0.0p+0f;case 477:return 0x0.0p+0f;case 478:return 0x0.0p+0f;case 479:return 0x0.0p+0f;case 480:return 0x0.0p+0f;case 481:return 0x0.0p+0f;case 482:return 0x0.0p+0f;case 483:return 0x0.0p+0f;case 484:return 0x0.0p+0f;case 485:return 0x0.0p+0f;case 486:return 0x0.0p+0f;case 487:return 0x0.0p+0f;case 488:return 0x0.0p+0f;case 489:return 0x0.0p+0f;case 490:return 0x0.0p+0f;case 491:return 0x0.0p+0f;case 492:return 0x0.0p+0f;case 493:return 0x0.0p+0f;case 494:return 0x0.0p+0f;case 495:return 0x0.0p+0f;case 496:return 0x0.0p+0f;case 497:return 0x0.0p+0f;case 498:return 0x0.0p+0f;case 499:return 0x0.0p+0f;case 500:return 0x0.0p+0f;case 501:return 0x0.0p+0f;case 502:return 0x0.0p+0f;case 503:return 0x0.0p+0f;case 504:return 0x0.0p+0f;case 505:return 0x0.0p+0f;case 506:return 0x0.0p+0f;case 507:return 0x0.0p+0f;case 508:return 0x0.0p+0f;case 509:return 0x0.0p+0f;case 510:return 0x0.0p+0f;case 511:return 0x0.0p+0f;case 512:return 0x0.0p+0f;case 513:return 0x0.0p+0f;case 514:return 0x0.0p+0f;case 515:return 0x0.0p+0f;case 516:return 0x0.0p+0f;case 517:return 0x0.0p+0f;case 518:return 0x0.0p+0f;case 519:return 0x0.0p+0f;case 520:return 0x0.0p+0f;case 521:return 0x0.0p+0f;case 522:return 0x0.0p+0f;case 523:return 0x0.0p+0f;case 524:return 0x0.0p+0f;case 525:return 0x0.0p+0f;case 526:return 0x0.0p+0f;case 527:return 0x0.0p+0f;case 528:return 0x0.0p+0f;case 529:return 0x0.0p+0f;case 530:return 0x0.0p+0f;case 531:return 0x0.0p+0f;case 532:return 0x0.0p+0f;case 533:return 0x0.0p+0f;case 534:return 0x0.0p+0f;case 535:return 0x0.0p+0f;case 536:return 0x0.0p+0f;case 537:return 0x0.0p+0f;case 538:return 0x0.0p+0f;case 539:return 0x0.0p+0f;case 540:return 0x0.0p+0f;case 541:return 0x0.0p+0f;case 542:return 0x0.0p+0f;case 543:return 0x0.0p+0f;case 544:return 0x0.0p+0f;case 545:return 0x0.0p+0f;case 546:return 0x0.0p+0f;case 547:return 0x0.0p+0f;case 548:return 0x0.0p+0f;case 549:return 0x0.0p+0f;case 550:return 0x0.0p+0f;case 551:return 0x0.0p+0f;case 552:return 0x0.0p+0f;case 553:return 0x0.0p+0f;case 554:return 0x0.0p+0f;case 555:return 0x0.0p+0f;case 556:return 0x0.0p+0f;case 557:return 0x0.0p+0f;case 558:return 0x0.0p+0f;case 559:return 0x0.0p+0f;case 560:return 0x0.0p+0f;case 561:return 0x0.0p+0f;case 562:return 0x0.0p+0f;case 563:return 0x0.0p+0f;case 564:return 0x0.0p+0f;case 565:return 0x0.0p+0f;case 566:return 0x0.0p+0f;case 567:return 0x0.0p+0f;case 568:return 0x0.0p+0f;case 569:return 0x0.0p+0f;case 570:return 0x0.0p+0f;case 571:return 0x0.0p+0f;case 572:return 0x0.0p+0f;case 573:return 0x0.0p+0f;case 574:return 0x0.0p+0f;case 575:return 0x0.0p+0f;case 576:return 0x0.0p+0f;case 577:return 0x0.0p+0f;case 578:return 0x0.0p+0f;case 579:return 0x0.0p+0f;case 580:return 0x0.0p+0f;case 581:return 0x0.0p+0f;case 582:return 0x0.0p+0f;case 583:return 0x0.0p+0f;case 584:return 0x0.0p+0f;case 585:return 0x0.0p+0f;case 586:return 0x0.0p+0f;case 587:return 0x0.0p+0f;case 588:return 0x0.0p+0f;case 589:return 0x0.0p+0f;case 590:return 0x0.0p+0f;case 591:return 0x0.0p+0f;case 592:return 0x0.0p+0f;case 593:return 0x0.0p+0f;case 594:return 0x0.0p+0f;case 595:return 0x0.0p+0f;case 596:return 0x0.0p+0f;case 597:return 0x0.0p+0f;case 598:return 0x0.0p+0f;case 599:return 0x0.0p+0f;case 600:return 0x0.0p+0f;case 601:return 0x0.0p+0f;case 602:return 0x0.0p+0f;case 603:return 0x0.0p+0f;case 604:return 0x0.0p+0f;case 605:return 0x0.0p+0f;case 606:return 0x0.0p+0f;case 607:return 0x0.0p+0f;case 608:return 0x0.0p+0f;case 609:return 0x0.0p+0f;case 610:return 0x0.0p+0f;case 611:return 0x0.0p+0f;case 612:return 0x0.0p+0f;case 613:return 0x0.0p+0f;case 614:return 0x0.0p+0f;case 615:return 0x0.0p+0f;case 616:return 0x0.0p+0f;case 617:return 0x0.0p+0f;case 618:return 0x0.0p+0f;case 619:return 0x0.0p+0f;case 620:return 0x0.0p+0f;case 621:return 0x0.0p+0f;case 622:return 0x0.0p+0f;case 623:return 0x0.0p+0f;case 624:return 0x0.0p+0f;case 625:return 0x0.0p+0f;case 626:return 0x0.0p+0f;case 627:return 0x0.0p+0f;case 628:return 0x0.0p+0f;case 629:return 0x0.0p+0f;case 630:return 0x0.0p+0f;case 631:return 0x0.0p+0f;case 632:return 0x0.0p+0f;case 633:return 0x0.0p+0f;case 634:return 0x0.0p+0f;case 635:return 0x0.0p+0f;case 636:return 0x0.0p+0f;case 637:return 0x0.0p+0f;case 638:return 0x0.0p+0f;case 639:return 0x0.0p+0f;case 640:return 0x0.0p+0f;case 641:return 0x0.0p+0f;case 642:return 0x0.0p+0f;case 643:return 0x0.0p+0f;case 644:return 0x0.0p+0f;case 645:return 0x0.0p+0f;case 646:return 0x0.0p+0f;case 647:return 0x0.0p+0f;case 648:return 0x0.0p+0f;case 649:return 0x0.0p+0f;case 650:return 0x0.0p+0f;case 651:return 0x0.0p+0f;case 652:return 0x0.0p+0f;case 653:return 0x0.0p+0f;case 654:return 0x0.0p+0f;case 655:return 0x0.0p+0f;case 656:return 0x0.0p+0f;case 657:return 0x0.0p+0f;case 658:return 0x0.0p+0f;case 659:return 0x0.0p+0f;case 660:return 0x0.0p+0f;case 661:return 0x0.0p+0f;case 662:return 0x0.0p+0f;case 663:return 0x0.0p+0f;case 664:return 0x0.0p+0f;case 665:return 0x0.0p+0f;case 666:return 0x0.0p+0f;case 667:return 0x0.0p+0f;case 668:return 0x0.0p+0f;case 669:return 0x0.0p+0f;case 670:return 0x0.0p+0f;case 671:return 0x0.0p+0f;case 672:return 0x0.0p+0f;case 673:return 0x0.0p+0f;case 674:return 0x0.0p+0f;case 675:return 0x0.0p+0f;case 676:return 0x0.0p+0f;case 677:return 0x0.0p+0f;case 678:return 0x0.0p+0f;case 679:return 0x0.0p+0f;case 680:return 0x0.0p+0f;case 681:return 0x0.0p+0f;case 682:return 0x0.0p+0f;case 683:return 0x0.0p+0f;case 684:return 0x0.0p+0f;case 685:return 0x0.0p+0f;case 686:return 0x0.0p+0f;case 687:return 0x0.0p+0f;case 688:return 0x0.0p+0f;case 689:return 0x0.0p+0f;case 690:return 0x0.0p+0f;case 691:return 0x0.0p+0f;case 692:return 0x0.0p+0f;case 693:return 0x0.0p+0f;case 694:return 0x0.0p+0f;case 695:return 0x0.0p+0f;case 696:return 0x0.0p+0f;case 697:return 0x0.0p+0f;case 698:return 0x0.0p+0f;case 699:return 0x0.0p+0f;case 700:return 0x0.0p+0f;case 701:return 0x0.0p+0f;case 702:return 0x0.0p+0f;case 703:return 0x0.0p+0f;case 704:return 0x0.0p+0f;case 705:return 0x0.0p+0f;case 706:return 0x0.0p+0f;case 707:return 0x0.0p+0f;case 708:return 0x0.0p+0f;case 709:return 0x0.0p+0f;case 710:return 0x0.0p+0f;case 711:return 0x0.0p+0f;case 712:return 0x0.0p+0f;case 713:return 0x0.0p+0f;case 714:return 0x0.0p+0f;case 715:return 0x0.0p+0f;case 716:return 0x0.0p+0f;case 717:return 0x0.0p+0f;case 718:return 0x0.0p+0f;case 719:return 0x0.0p+0f;case 720:return 0x0.0p+0f;case 721:return 0x0.0p+0f;case 722:return 0x0.0p+0f;case 723:return 0x0.0p+0f;case 724:return 0x0.0p+0f;case 725:return 0x0.0p+0f;case 726:return 0x0.0p+0f;case 727:return 0x0.0p+0f;case 728:return 0x0.0p+0f;case 729:return 0x0.0p+0f;case 730:return 0x0.0p+0f;case 731:return 0x0.0p+0f;case 732:return 0x0.0p+0f;case 733:return 0x0.0p+0f;case 734:return 0x0.0p+0f;case 735:return 0x0.0p+0f;case 736:return 0x0.0p+0f;case 737:return 0x0.0p+0f;case 738:return 0x0.0p+0f;case 739:return 0x0.0p+0f;case 740:return 0x0.0p+0f;case 741:return 0x0.0p+0f;case 742:return 0x0.0p+0f;case 743:return 0x0.0p+0f;case 744:return 0x0.0p+0f;case 745:return 0x0.0p+0f;case 746:return 0x0.0p+0f;case 747:return 0x0.0p+0f;case 748:return 0x0.0p+0f;case 749:return 0x0.0p+0f;case 750:return 0x0.0p+0f;case 751:return 0x0.0p+0f;case 752:return 0x0.0p+0f;case 753:return 0x0.0p+0f;case 754:return 0x0.0p+0f;case 755:return 0x0.0p+0f;case 756:return 0x0.0p+0f;case 757:return 0x0.0p+0f;case 758:return 0x0.0p+0f;case 759:return 0x0.0p+0f;case 760:return 0x0.0p+0f;case 761:return 0x0.0p+0f;case 762:return 0x0.0p+0f;case 763:return 0x0.0p+0f;case 764:return 0x0.0p+0f;case 765:return 0x0.0p+0f;case 766:return 0x0.0p+0f;case 767:return 0x0.0p+0f;case 768:return 0x0.0p+0f;case 769:return 0x0.0p+0f;case 770:return 0x0.0p+0f;case 771:return 0x0.0p+0f;case 772:return 0x0.0p+0f;case 773:return 0x0.0p+0f;case 774:return 0x0.0p+0f;case 775:return 0x0.0p+0f;case 776:return 0x0.0p+0f;case 777:return 0x0.0p+0f;case 778:return 0x0.0p+0f;case 779:return 0x0.0p+0f;case 780:return 0x0.0p+0f;case 781:return 0x0.0p+0f;case 782:return 0x0.0p+0f;case 783:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_dnu { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.0000000000000p+1f;case 1:return 0x0.0p+0f;case 2:return 0x0.0p+0f;case 3:return 0x0.0p+0f;case 4:return 0x0.0p+0f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_fx { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x0.0p+0f;case 1:return 0x1.0ef43a0000000p+7f;case 2:return 0x0.0p+0f;case 3:return 0x0.0p+0f;case 4:return 0x0.0p+0f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_ido { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 1;case 1:return 1;case 2:return 1;case 3:return 1;case 4:return 1;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;case 26:return 0;case 27:return 0;default: __builtin_unreachable();}}};
struct coeff_ipc { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 2;case 1:return 3;case 2:return 1;case 3:return 4;case 4:return 5;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;case 26:return 0;case 27:return 0;default: __builtin_unreachable();}}};
struct coeff_lliq { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(li))*(1)) {case 0:return 0;case 1:return 0;case 2:return 0;case 3:return 0;case 4:return 0;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;default: __builtin_unreachable();}}};
struct coeff_ln { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0;case 1:return 0;case 2:return 0;case 3:return 0;case 4:return 0;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;case 26:return 0;case 27:return 0;default: __builtin_unreachable();}}};
struct coeff_lvol { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0;case 1:return 0;case 2:return 0;case 3:return 0;case 4:return 0;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;case 26:return 0;case 27:return 0;default: __builtin_unreachable();}}};
struct coeff_lz { __device__ __forceinline__ int operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0;case 1:return 0;case 2:return 0;case 3:return 0;case 4:return 0;case 5:return 0;case 6:return 0;case 7:return 0;case 8:return 0;case 9:return 0;case 10:return 0;case 11:return 0;case 12:return 0;case 13:return 0;case 14:return 0;case 15:return 0;case 16:return 0;case 17:return 0;case 18:return 0;case 19:return 0;case 20:return 0;case 21:return 0;case 22:return 0;case 23:return 0;case 24:return 0;case 25:return 0;case 26:return 0;case 27:return 0;default: __builtin_unreachable();}}};
struct coeff_mmgraupvt { __device__ __forceinline__ float operator()(int i0,int i1) const { switch(0+(i0-(1))*(1)+(i1-(1))*(1*((ngdnmm)-(1)+1))) {case 0:return 0x1.9000000000000p+5f;case 1:return 0x1.2c00000000000p+7f;case 2:return 0x1.f400000000000p+7f;case 3:return 0x1.5e00000000000p+8f;case 4:return 0x1.c200000000000p+8f;case 5:return 0x1.1300000000000p+9f;case 6:return 0x1.4500000000000p+9f;case 7:return 0x1.7700000000000p+9f;case 8:return 0x1.a900000000000p+9f;case 9:return 0x1.f7624e0000000p+5f;case 10:return 0x1.787cee0000000p+6f;case 11:return 0x1.caf5c20000000p+6f;case 12:return 0x1.066b860000000p+7f;case 13:return 0x1.22851e0000000p+7f;case 14:return 0x1.3b6b860000000p+7f;case 15:return 0x1.51f5c20000000p+7f;case 16:return 0x1.66b8520000000p+7f;case 17:return 0x1.7a0a3e0000000p+7f;case 18:return 0x1.5b3bb80000000p-1f;case 19:return 0x1.4699840000000p-1f;case 20:return 0x1.3e72da0000000p-1f;case 21:return 0x1.398c7e0000000p-1f;case 22:return 0x1.3620ee0000000p-1f;case 23:return 0x1.3389b60000000p-1f;case 24:return 0x1.31797c0000000p-1f;case 25:return 0x1.2fc5040000000p-1f;case 26:return 0x1.2e53660000000p-1f;default: __builtin_unreachable();}}};
struct coeff_qxmin { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.12e0be0000000p-30f;case 1:return 0x1.ad7f2a0000000p-24f;case 2:return 0x1.1979980000000p-40f;case 3:return 0x1.ad7f2a0000000p-24f;case 4:return 0x1.ad7f2a0000000p-24f;case 5:return 0x1.1979980000000p-40f;case 6:return 0x1.1979980000000p-40f;case 7:return 0x1.1979980000000p-40f;case 8:return 0x1.1979980000000p-40f;case 9:return 0x1.1979980000000p-40f;case 10:return 0x1.1979980000000p-40f;case 11:return 0x1.1979980000000p-40f;case 12:return 0x1.1979980000000p-40f;case 13:return 0x1.1979980000000p-40f;case 14:return 0x1.1979980000000p-40f;case 15:return 0x1.1979980000000p-40f;case 16:return 0x1.1979980000000p-40f;case 17:return 0x1.1979980000000p-40f;case 18:return 0x1.1979980000000p-40f;case 19:return 0x1.1979980000000p-40f;case 20:return 0x1.1979980000000p-40f;case 21:return 0x1.1979980000000p-40f;case 22:return 0x1.1979980000000p-40f;case 23:return 0x1.1979980000000p-40f;case 24:return 0x1.1979980000000p-40f;case 25:return 0x1.1979980000000p-40f;case 26:return 0x1.1979980000000p-40f;case 27:return 0x1.1979980000000p-40f;default: __builtin_unreachable();}}};
struct coeff_xdn0 { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.f400000000000p+9f;case 1:return 0x1.f400000000000p+9f;case 2:return 0x1.c200000000000p+9f;case 3:return 0x1.9000000000000p+6f;case 4:return 0x1.f400000000000p+8f;case 5:return 0x1.c200000000000p+9f;case 6:return 0x1.c200000000000p+9f;case 7:return 0x1.c200000000000p+9f;case 8:return 0x1.c200000000000p+9f;case 9:return 0x1.c200000000000p+9f;case 10:return 0x1.c200000000000p+9f;case 11:return 0x1.c200000000000p+9f;case 12:return 0x1.c200000000000p+9f;case 13:return 0x1.c200000000000p+9f;case 14:return 0x1.c200000000000p+9f;case 15:return 0x1.c200000000000p+9f;case 16:return 0x1.c200000000000p+9f;case 17:return 0x1.c200000000000p+9f;case 18:return 0x1.c200000000000p+9f;case 19:return 0x1.c200000000000p+9f;case 20:return 0x1.c200000000000p+9f;case 21:return 0x1.c200000000000p+9f;case 22:return 0x1.c200000000000p+9f;case 23:return 0x1.c200000000000p+9f;case 24:return 0x1.c200000000000p+9f;case 25:return 0x1.c200000000000p+9f;case 26:return 0x1.c200000000000p+9f;case 27:return 0x1.c200000000000p+9f;default: __builtin_unreachable();}}};
struct coeff_xdnmn { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.f400000000000p+9f;case 1:return 0x1.f400000000000p+9f;case 2:return 0x1.9000000000000p+6f;case 3:return 0x1.9000000000000p+6f;case 4:return 0x1.5400000000000p+7f;case 5:return 0x1.c200000000000p+9f;case 6:return 0x1.c200000000000p+9f;case 7:return 0x1.c200000000000p+9f;case 8:return 0x1.c200000000000p+9f;case 9:return 0x1.c200000000000p+9f;case 10:return 0x1.c200000000000p+9f;case 11:return 0x1.c200000000000p+9f;case 12:return 0x1.c200000000000p+9f;case 13:return 0x1.c200000000000p+9f;case 14:return 0x1.c200000000000p+9f;case 15:return 0x1.c200000000000p+9f;case 16:return 0x1.c200000000000p+9f;case 17:return 0x1.c200000000000p+9f;case 18:return 0x1.c200000000000p+9f;case 19:return 0x1.c200000000000p+9f;case 20:return 0x1.c200000000000p+9f;case 21:return 0x1.c200000000000p+9f;case 22:return 0x1.c200000000000p+9f;case 23:return 0x1.c200000000000p+9f;case 24:return 0x1.c200000000000p+9f;case 25:return 0x1.c200000000000p+9f;case 26:return 0x1.c200000000000p+9f;case 27:return 0x1.c200000000000p+9f;default: __builtin_unreachable();}}};
struct coeff_xdnmx { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.f400000000000p+9f;case 1:return 0x1.f400000000000p+9f;case 2:return 0x1.ca80000000000p+9f;case 3:return 0x1.2c00000000000p+8f;case 4:return 0x1.c200000000000p+9f;case 5:return 0x1.c200000000000p+9f;case 6:return 0x1.c200000000000p+9f;case 7:return 0x1.c200000000000p+9f;case 8:return 0x1.c200000000000p+9f;case 9:return 0x1.c200000000000p+9f;case 10:return 0x1.c200000000000p+9f;case 11:return 0x1.c200000000000p+9f;case 12:return 0x1.c200000000000p+9f;case 13:return 0x1.c200000000000p+9f;case 14:return 0x1.c200000000000p+9f;case 15:return 0x1.c200000000000p+9f;case 16:return 0x1.c200000000000p+9f;case 17:return 0x1.c200000000000p+9f;case 18:return 0x1.c200000000000p+9f;case 19:return 0x1.c200000000000p+9f;case 20:return 0x1.c200000000000p+9f;case 21:return 0x1.c200000000000p+9f;case 22:return 0x1.c200000000000p+9f;case 23:return 0x1.c200000000000p+9f;case 24:return 0x1.c200000000000p+9f;case 25:return 0x1.c200000000000p+9f;case 26:return 0x1.c200000000000p+9f;case 27:return 0x1.c200000000000p+9f;default: __builtin_unreachable();}}};
struct coeff_xmu { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.0000000000000p+0f;case 1:return 0x1.5555560000000p-2f;case 2:return 0x1.0000000000000p+0f;case 3:return 0x1.5555560000000p-2f;case 4:return 0x1.5555560000000p-2f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_xnu { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x0.0p+0f;case 1:return -0x1.5555560000000p-1f;case 2:return 0x0.0p+0f;case 3:return -0x1.5555560000000p-1f;case 4:return -0x1.5555560000000p-1f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_xvmn { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.35140c0000000p-55f;case 1:return 0x1.2dd5920000000p-42f;case 2:return 0x1.2dd5920000000p-51f;case 3:return 0x1.2dd5920000000p-51f;case 4:return 0x1.f1685a0000000p-37f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
struct coeff_xvmx { __device__ __forceinline__ float operator()(int i0) const { switch(0+(i0-(lc))*(1)) {case 0:return 0x1.fd58680000000p-41f;case 1:return 0x1.e5bfe60000000p-24f;case 2:return 0x1.1fda060000000p-28f;case 3:return 0x1.191ae40000000p-21f;case 4:return 0x1.191ae40000000p-18f;case 5:return 0x0.0p+0f;case 6:return 0x0.0p+0f;case 7:return 0x0.0p+0f;case 8:return 0x0.0p+0f;case 9:return 0x0.0p+0f;case 10:return 0x0.0p+0f;case 11:return 0x0.0p+0f;case 12:return 0x0.0p+0f;case 13:return 0x0.0p+0f;case 14:return 0x0.0p+0f;case 15:return 0x0.0p+0f;case 16:return 0x0.0p+0f;case 17:return 0x0.0p+0f;case 18:return 0x0.0p+0f;case 19:return 0x0.0p+0f;case 20:return 0x0.0p+0f;case 21:return 0x0.0p+0f;case 22:return 0x0.0p+0f;case 23:return 0x0.0p+0f;case 24:return 0x0.0p+0f;case 25:return 0x0.0p+0f;case 26:return 0x0.0p+0f;case 27:return 0x0.0p+0f;default: __builtin_unreachable();}}};
static constexpr int ac_opt=0;
static constexpr float advisc0=0x1.335bce0000000p-16f;
static constexpr float advisc1=0x1.203b8a0000000p-16f;
static constexpr float alphah=0x0.0p+0f;
static constexpr float alphahacx=0x0.0p+0f;
static constexpr float alphahl=0x1.0000000000000p+0f;
static constexpr float alphamax=0x1.e000000000000p+3f;
static constexpr float alphamin=0x0.0p+0f;
static constexpr float alphar=0x0.0p+0f;
static constexpr float alpharaut=0x0.0p+0f;
static constexpr float alpharmax=0x1.0000000000000p+3f;
static constexpr float alphas=0x0.0p+0f;
static constexpr float alphasmlr0=0x1.c000000000000p+3f;
float ar=0x1.a4ff920000000p+9f;
float aradcw=-0x1.1a0cf20000000p-2f;
Array<const float> ax{init_ax,lc,lqmx};
Array<const float> bb{init_bb,lc,lqmx};
static constexpr float bfnu0=0x1.8000020000000p+2f;
static constexpr float biggsnowdiam=-0x1.0000000000000p+0f;
float br=0x1.99999a0000000p-1f;
float bradcw=0x1.0056800000000p+18f;
float bta1=0x1.3333340000000p-1f;
Array<const float> bx{init_bx,lc,lqmx};
static constexpr float c1sw=0x1.3d714e0000000p-1f;
static constexpr float cai=0x1.5dfe280000000p+4f;
static constexpr float caw=0x1.144f6a0000000p+4f;
static constexpr float cbi=0x1.ea3d700000000p+2f;
static constexpr float cbw=0x1.1ee1480000000p+5f;
static constexpr float ccn=0x1.dcd6500000000p+28f;
static constexpr float ccntimeconst=0x1.c200000000000p+11f;
static constexpr float cdhdnmax=0x1.9000000000000p+9f;
static constexpr float cdhdnmin=0x1.f400000000000p+8f;
static constexpr float cdhldnmax=0x1.9000000000000p+9f;
static constexpr float cdhldnmin=0x1.f400000000000p+8f;
static constexpr float cdhlmax=0x1.3333340000000p-1f;
static constexpr float cdhlmin=0x1.cccccc0000000p-2f;
static constexpr float cdhmax=0x1.99999a0000000p-1f;
static constexpr float cdhmin=0x1.cccccc0000000p-2f;
Array<const float> cdx{init_cdx,lc,lqmx};
static constexpr float cfnfac=0x1.99999a0000000p-4f;
struct ciacrratio_view { __device__ __forceinline__ float operator()(int i,int j) const { return ::g_nssl2_gs_tables.ciacrratio[i][j]; } }; ciacrratio_view ciacrratio;
static constexpr float ciintmx=0x1.e848000000000p+19f;
static constexpr float cimas0=0x1.2326920000000p-34f;
static constexpr float cimas1=0x1.834f400000000p-41f;
static constexpr float cimn=0x1.f400000000000p+9f;
static constexpr float cimx=0x1.e848000000000p+19f;
static constexpr float cinu=0x0.0p+0f;
float cnit=0x1.47ae140000000p-7f;
Array<const float> cno{init_cno,lc,lqmx};
float cnoh=0x1.86a0000000000p+18f;
float cnohl=0x1.3880000000000p+15f;
static constexpr float cnohlmn=0x1.47ae140000000p-7f;
static constexpr float cnohmn=0x1.47ae140000000p-7f;
float cnor=0x1.86a0000000000p+19f;
float cnos=0x1.6e36000000000p+21f;
static constexpr float cp=0x1.f600000000000p+9f;
static constexpr float cpi=0x1.0519800000000p-10f;
static constexpr float cpigb=0x1.0740000000000p+11f;
static constexpr float cpl=0x1.05e0000000000p+12f;
float cradcw=-0x1.1992840000000p+34f;
float cs=0x1.8d70a40000000p+3f;
static constexpr float cv=0x1.6680000000000p+9f;
static constexpr float cvv=0x1.6020000000000p+10f;
float cwc1=0x1.f4a87c0000000p-10f;
static constexpr float cwccn=0x1.dcd6500000000p+28f;
static constexpr float cwdiap=0x1.4f8b580000000p-16f;
static constexpr float cwdisp=0x1.3333340000000p-3f;
static constexpr float cwfrz2snowfrac=0x0.0p+0f;
static constexpr float cwfrz2snowratio=0x1.4000000000000p+2f;
static constexpr float cwmasn=0x1.2dd5940000000p-45f;
static constexpr float cwmasx=0x1.f168560000000p-31f;
static constexpr float cwradn=0x1.0c6f7a0000000p-19f;
static constexpr float cwradx=0x1.f751040000000p-15f;
static constexpr float cxmin=0x1.5798ee0000000p-27f;
Array<const float> da0{init_da0,lc,lqmx};
Array<const float> da1{init_da1,lc,lqmx};
Array<const float> dab0{init_dab0,lc,lqmx,lc,lqmx};
struct dab0lu_view { __device__ __forceinline__ float operator()(int,int,int,int) const { return 0.0f; } }; dab0lu_view dab0lu;
Array<const float> dab1{init_dab1,lc,lqmx,lc,lqmx};
struct dab1lu_view { __device__ __forceinline__ float operator()(int,int,int,int) const { return 0.0f; } }; dab1lu_view dab1lu;
static constexpr float depfac=0x1.0000000000000p+0f;
static constexpr float dfrz=0x1.3a92a40000000p-13f;
static constexpr float dg0thresh=0x1.3333340000000p-3f;
static constexpr double dgam=0x1.47ae140000000p-7;
static constexpr double dgami=0x1.9000000000000p+6;
static constexpr float dhlmn=0x1.3a92a40000000p-12f;
static constexpr float dhlmx=0x1.47ae140000000p-5f;
static constexpr float dhmn=0x1.3a92a40000000p-12f;
static constexpr float dhmn0=0x1.3a92a40000000p-12f;
static constexpr float dimax=0x1.0624de0000000p-11f;
static constexpr int dmhlopt=0;
static constexpr float dmincw=0x1.f751040000000p-17f;
static constexpr float dmlt=0x1.89374c0000000p-9f;
static constexpr int dmrauto=0;
static constexpr int dmropt=0;
static constexpr float dmuh=0x1.0000000000000p+0f;
static constexpr float dmuhl=0x1.0000000000000p+0f;
Array<const float> dnu{init_dnu,lc,lqmx};
float dnz00=0x1.39999a0000000p+0f;
static constexpr bool do_accurate_sedimentation=false;
static constexpr float dqiacralpha=0x1.99999a0000000p-5f;
static constexpr float dqiacralphainv=0x1.4000000000000p+4f;
static constexpr float dqiacrratio=0x1.0000000000000p-2f;
static constexpr float dqiacrratioinv=0x1.0000000000000p+2f;
float dradcw=0x1.95df000000000p+48f;
float dragh=0x1.3333340000000p-1f;
float ds=0x1.ae147a0000000p-2f;
static constexpr float dshd=0x1.0624de0000000p-10f;
static constexpr float dwehwmin=0x0.0p+0f;
static constexpr float dwmax=0x1.eb851e0000000p-7f;
static constexpr float dwmin=0x1.47ae140000000p-8f;
static constexpr float dwtempmin=0x1.e400000000000p+7f;
static constexpr float ehimax=0x1.0000000000000p+0f;
static constexpr float ehimin=0x0.0p+0f;
static constexpr float ehlr0=0x1.0000000000000p+0f;
static constexpr float ehlw0=0x1.cccccc0000000p-1f;
static constexpr float ehr0=0x1.0000000000000p+0f;
static constexpr float ehs0=0x1.99999a0000000p-4f;
static constexpr float ehs1=0x1.99999a0000000p-4f;
static constexpr float ehsmax=0x1.0000000000000p-1f;
static constexpr float ehw0=0x1.cccccc0000000p-1f;
static constexpr float eii0=0x1.99999a0000000p-4f;
static constexpr float eii0hl=0x1.99999a0000000p-3f;
static constexpr float eii1=0x1.99999a0000000p-4f;
static constexpr float eii1hl=0x0.0p+0f;
static constexpr float eqtot=0x1.12e0be0000000p-30f;
static constexpr int eqtset=1;
static constexpr float eri0=0x1.99999a0000000p-4f;
static constexpr float eri_cimin=0x1.4f8b580000000p-17f;
static constexpr float esi0=0x1.99999a0000000p-4f;
static constexpr float ess0=0x1.0000000000000p-1f;
static constexpr float ess1=0x1.99999a0000000p-5f;
static constexpr float essfrac1=0x1.0000000000000p-1f;
static constexpr float essfrac2=0x1.8000000000000p-1f;
static constexpr float essrmax=0x1.47ae140000000p-6f;
static constexpr float esstem1=-0x1.e000000000000p+3f;
static constexpr float esstem2=-0x1.4000000000000p+3f;
static constexpr float evapfac=0x1.0000000000000p+0f;
static constexpr float ewfac=0x1.0000000000000p+0f;
static constexpr float ewi_dcmin=0x1.f751040000000p-17f;
static constexpr float ewi_dimin=0x1.f751040000000p-16f;
static constexpr float exwmindiam=0x0.0p+0f;
static constexpr float f2h=0x1.0000000000000p+0f;
static constexpr float ffrzs=0x0.0p+0f;
static constexpr float fqsat=0x1.0624de0000000p-9f;
static constexpr float fscni=0x1.0000000000000p+0f;
Array<const float> fx{init_fx,lc,lqmx};
float gf4br=0x1.1d67e80000000p+4f;
float gf4ds=0x1.4d57240000000p+3f;
float gf4p5=0x1.74371e0000000p+3f;
static constexpr float gfcinu1=0x1.0000000000000p+0f;
static constexpr float gfcinu1p18=0x1.d874e80000000p-1f;
static constexpr float gfcinu1p22=0x1.d35dc40000000p-1f;
static constexpr float gfcinu1p47=0x1.c5736c0000000p-1f;
static constexpr float gfcinu2p18=0x1.17893e0000000p+0f;
static constexpr float gfcinu2p22=0x1.1d5de80000000p+0f;
static constexpr float gfcinu2p47=0x1.4daa3e0000000p+0f;
struct gmoi_view { __device__ __forceinline__ auto operator()(int i) const { return ::nssl2_gmoi(i); } }; gmoi_view gmoi;
float gr=0x1.39999a0000000p+3f;
static constexpr float graupelfallfac=0x1.0000000000000p+0f;
static constexpr float gsnow1=0x1.25d0640000000p+2f;
static constexpr float gsnow73=0x1.c688bc0000000p-1f;
static constexpr float hailfallfac=0x1.0000000000000p+0f;
static constexpr float hdnmn=0x1.5400000000000p+7f;
static constexpr float hlcnhdia=0x1.0624de0000000p-10f;
static constexpr float hlcnhqmin=0x1.a36e2e0000000p-14f;
static constexpr float hldia1=0x1.47ae140000000p-7f;
float hldnmn=0x1.f400000000000p+8f;
static constexpr int hlssflg=1;
static constexpr int hssflg=1;
static constexpr int iacr=2;
static constexpr int iacrsize=5;
static constexpr int ialpstart=-19;
static constexpr int ibfc=1;
static constexpr int ibfr=2;
static constexpr int ibiggopt=2;
static constexpr int ibiggsmallrain=0;
static constexpr int ibiggsnow=3;
int ibinhlmlr=0;
int ibinhmlr=0;
static constexpr int ibinnum=2;
static constexpr int icdx=6;
static constexpr int icdxhl=6;
static constexpr float icefallfac=0x1.0000000000000p+0f;
static constexpr int icefallopt=3;
static constexpr int icenucopt=1;
static constexpr int icespheres=0;
static constexpr int icfn=2;
static constexpr int icond=1;
static constexpr int icracr=1;
static constexpr int icracrthresh=1;
static constexpr int icrcev=1;
static constexpr int icvhl2h=0;
static constexpr int idbzci=1;
static constexpr int idiagnosecnu=0;
Array<const int> ido{init_ido,lc,lqmx};
static constexpr int iehlr0c=0;
int iehlw=1;
static constexpr int iehr0c=0;
static constexpr int iehw=1;
static constexpr int iessec0flag=0;
static constexpr int iessopt=1;
static constexpr int iferwisventr=2;
static constexpr float ifiacrg=0x1.0000000000000p+0f;
static constexpr float ifrzg=0x1.0000000000000p+0f;
static constexpr float ifrzs=0x1.0000000000000p+0f;
static constexpr int iglcnvi=1;
static constexpr int iglcnvs=2;
static constexpr int ihlcnh=0;
static constexpr int ihmlt=2;
static constexpr int ihrn=0;
static constexpr int ilimit=0;
static constexpr int imaxdiaopt=3;
static constexpr bool imeyers5=false;
static constexpr int imixedphase=0;
static constexpr int imltshddmr=1;
static constexpr int imurain=1;
static constexpr int imusnow=3;
static constexpr int imydiagalpha=0;
static constexpr int incwet=0;
static constexpr int infall=4;
static constexpr int interval_sedi_vt=1;
static constexpr bool invertccn=false;
static constexpr int ioldlimiter=0;
Array<const int> ipc{init_ipc,lc,lqmx};
static constexpr int ipconc=0;
static constexpr int ipelec=0;
static constexpr int iqhacrmlr=1;
static constexpr int iqhacwshr=1;
static constexpr int iqhlacrmlr=1;
static constexpr int iqhlacwshr=1;
static constexpr int iqis0=2;
static constexpr int ircnw=5;
static constexpr int irenuc=2;
static constexpr int irescalerainopt=3;
static constexpr int iresetmoments=0;
static constexpr int irfall=4;
static constexpr int irimdenopt=1;
static constexpr int irwfrz=1;
static constexpr int iscni=4;
static constexpr int isfall=2;
static constexpr int isnowdens=1;
static constexpr int isnowfall=2;
static constexpr int isnwfrac=0;
static constexpr int itype1=0;
static constexpr int itype2=2;
static constexpr int iusedw=0;
static constexpr int iusewetgraupel=1;
static constexpr int iusewethail=0;
static constexpr int iusewetsnow=0;
static constexpr int ivhmltsoak=1;
static constexpr int ivshdgs=1;
static constexpr bool iwetsoak=true;
static constexpr int ixtaltype=1;
static constexpr int izwisventr=2;
static constexpr float lawson_splinter_fac=0x1.b7cdfe0000000p-36f;
static constexpr int lc=3;
static constexpr int lccn=0;
static constexpr int lccna=0;
static constexpr int lcin=0;
static constexpr int lcina=0;
bool ldovol=false;
static constexpr int lf=0;
static constexpr int lg=7;
static constexpr int lh=7;
static constexpr int lhab=7;
static constexpr int lhl=0;
static constexpr int lhlw=0;
static constexpr int lhlwlg=0;
static constexpr int lhw=0;
static constexpr int lhwlg=0;
static constexpr int li=5;
static constexpr int lis=0;
Array<const int> lliq{init_lliq,li,lqmx};
Array<const int> ln{init_ln,lc,lqmx};
static constexpr int lnc=0;
static constexpr int lnh=0;
static constexpr int lnhf=0;
static constexpr int lnhl=0;
static constexpr int lnhlf=0;
static constexpr int lni=0;
static constexpr int lnis=0;
static constexpr int lnr=0;
static constexpr int lns=0;
static constexpr int lqmx=30;
static constexpr int lr=4;
static constexpr int ls=6;
static constexpr int lss=0;
static constexpr int lsw=0;
static constexpr int lt=1;
static constexpr int lv=2;
static constexpr int lvh=0;
static constexpr int lvhl=0;
Array<const int> lvol{init_lvol,lc,lqmx};
static constexpr int lvs=0;
static constexpr bool lwsm6=false;
Array<const int> lz{init_lz,lc,lqmx};
static constexpr int lzh=0;
static constexpr int lzhl=0;
static constexpr int lzr=0;
static constexpr int lzs=0;
static constexpr float maxalphalu=0x1.e000000000000p+3f;
static constexpr float maxratiolu=0x1.9000000000000p+6f;
static constexpr float meltfac=0x1.0000000000000p+0f;
static constexpr float minalphalu=-0x1.e666660000000p-1f;
static constexpr bool mixedphase=false;
static constexpr float mltdiam1=0x1.26e9780000000p-7f;
static constexpr float mltdiam2=0x1.0624de0000000p-6f;
static constexpr float mltdiam3=0x1.374bc60000000p-6f;
static constexpr float mltdiam4=0x1.99999a0000000p-3f;
Array<const float> mmgraupvt{init_mmgraupvt,1,ngdnmm,1,3};
static constexpr int na=7;
static constexpr int ndebug=-1;
static constexpr int ngdnmm=9;
static constexpr int ngm0=3001;
static constexpr int nqiacralpha=300;
static constexpr int nqiacrratio=400;
static constexpr int nqsat=1000001;
static constexpr int nsplinter=0;
static constexpr int numshedregimes=3;
float pi=0x1.921fb60000000p+1f;
float pid4=0x1.921fb60000000p-1f;
static constexpr float piinv=0x1.45f3060000000p-2f;
static constexpr float qccn=0x1.85413c0000000p+28f;
static constexpr float qhacidn=0x1.5400000000000p+7f;
static constexpr float qhdpvdn=0x1.5400000000000p+7f;
struct qiacrratio_view { __device__ __forceinline__ float operator()(int i,int j) const { return ::g_nssl2_gs_tables.qiacrratio[i][j]; } }; qiacrratio_view qiacrratio;
static constexpr float qminrncw=0x1.0624de0000000p-9f;
static constexpr bool qsvtmod=false;
Array<const float> qxmin{init_qxmin,lc,lqmx};
static constexpr float rainfallfac=0x1.0000000000000p+0f;
static constexpr int rcond=2;
float rd=0x1.1f0a3e0000000p+8f;
static constexpr bool rescale_high_alpha=false;
static constexpr bool rescale_low_alpha=true;
static constexpr bool rescale_low_alphah=true;
static constexpr bool rescale_low_alphahl=true;
static constexpr bool rescale_low_alphar=true;
static constexpr float rescale_tempthresh=0x0.0p+0f;
static constexpr float rescale_wthresh=0x1.8000000000000p+1f;
static constexpr bool restoreccn=true;
static constexpr float rho00=0x1.39999a0000000p+0f;
static constexpr float rho_qh=0x1.f400000000000p+8f;
static constexpr float rho_qhl=0x1.c200000000000p+9f;
static constexpr float rho_qs=0x1.9000000000000p+6f;
static constexpr float rhofrz=0x1.c200000000000p+9f;
static constexpr float rimc1=0x1.2c00000000000p+8f;
static constexpr float rimc2=0x1.c28f5c0000000p-2f;
static constexpr float rimc3=0x1.5400000000000p+7f;
static constexpr float rimc4=0x1.c200000000000p+9f;
static constexpr float rimdenvwgt=0x0.0p+0f;
static constexpr float rnu=-0x1.99999a0000000p-1f;
static constexpr float rnumax=0x1.e000000000000p+3f;
static constexpr float rnumin=-0x1.99999a0000000p-1f;
static constexpr float roqimax=0x1.54c9900000000p-14f;
static constexpr float rovcp=0x1.24c2080000000p-2f;
static constexpr int rssflg=1;
static constexpr float rw=0x1.cd80000000000p+8f;
static constexpr float rz=0x1.0000000000000p+0f;
static constexpr float rzhl=0x1.c000000000000p-2f;
static constexpr float rzs=0x1.3333360000000p-2f;
static constexpr float sheddiam=0x1.0624de0000000p-7f;
static constexpr float sheddiam0=0x1.47ae140000000p-6f;
static constexpr float snowfallfac=0x1.0000000000000p+0f;
static constexpr float snowmeltdia=0x0.0p+0f;
static constexpr float snu=-0x1.99999a0000000p-1f;
static constexpr float splintermass=0x1.834f400000000p-41f;
static constexpr int sssflg=1;
struct tabqis_view { __device__ __forceinline__ auto operator()(int i) const { return ::tabqis(i); } }; tabqis_view tabqis;
struct tabqvs_view { __device__ __forceinline__ auto operator()(int i) const { return ::tabqvs(i); } }; tabqvs_view tabqvs;
static constexpr float takshedsize1=0x1.3333340000000p-3f;
static constexpr float takshedsize2=0x1.3333340000000p-2f;
static constexpr float takshedsize3=0x1.cccccc0000000p-2f;
float tfr=0x1.1126660000000p+8f;
float tfrcbi=0x1.097d700000000p+8f;
float tfrcbw=0x1.da947a0000000p+7f;
float tfrh=0x1.d24ccc0000000p+7f;
static constexpr float tka0=0x1.8e21960000000p-6f;
static constexpr float ventr=0x1.0000000000000p+0f;
static constexpr float ventrn=0x1.d3cd8e0000000p+0f;
static constexpr float vtmaxsed=0x1.1800000000000p+6f;
static constexpr float warmonly=0x0.0p+0f;
static constexpr float xcradmx=0x1.f751040000000p-15f;
Array<const float> xdn0{init_xdn0,lc,lqmx};
Array<const float> xdnmn{init_xdnmn,lc,lqmx};
Array<const float> xdnmx{init_xdnmx,lc,lqmx};
Array<const float> xmu{init_xmu,lc,lqmx};
Array<const float> xnu{init_xnu,lc,lqmx};
static constexpr float xvcmn=0x1.35140c0000000p-55f;
static constexpr float xvcmx=0x1.fd58680000000p-41f;
static constexpr float xvhlmn=0x1.f1685c0000000p-37f;
static constexpr float xvhlmx=0x1.191ae40000000p-15f;
static constexpr float xvhmn=0x1.f1685a0000000p-37f;
static constexpr float xvhmx=0x1.191ae40000000p-18f;
Array<const float> xvmn{init_xvmn,lc,lqmx};
Array<const float> xvmx{init_xvmx,lc,lqmx};
static constexpr float xvsmn=0x1.2dd5920000000p-51f;
static constexpr float xvsmx=0x1.191ae40000000p-21f;
struct ziacrratio_view { __device__ __forceinline__ float operator()(int i,int j) const { asm("trap;"); return 0.f; /* ipconc>=6 only; ipconc=0 image */ } }; ziacrratio_view ziacrratio;
static constexpr float zxmin=0x1.fb0f6c0000000p-94f;
__device__ __forceinline__ float gaml02(float x) {
int ig;
int i;
int ii;
int n;
int np;
constexpr int ng=12;
FixedStorage<float,(ng-1+1),1,ng> gamxg{};
FixedStorage<float,(ng-1+1),1,ng> xg{};
float _result;
// WRF:3623
xg.data[0]=0.01f;xg.data[1]=0.02f;xg.data[2]=0.025f;xg.data[3]=0.04f;xg.data[4]=0.075f;xg.data[5]=0.1f;xg.data[6]=0.25f;xg.data[7]=0.5f;xg.data[8]=0.75f;xg.data[9]=1.f;xg.data[10]=2.f;xg.data[11]=10.f;
// WRF:3624
gamxg.data[0]=7.391019203578037e-8f;gamxg.data[1]=0.02212726874591478f;gamxg.data[2]=0.06959352407989682f;gamxg.data[3]=0.2355654024970809f;gamxg.data[4]=0.46135930387500346f;gamxg.data[5]=0.545435791452399f;gamxg.data[6]=0.7371571313308203f;gamxg.data[7]=0.8265676632204345f;gamxg.data[8]=0.8640182781845841f;gamxg.data[9]=0.8855756211304151f;gamxg.data[10]=0.9245079225301251f;gamxg.data[11]=0.9712578342732681f;
// WRF:3631
if((x>=xg(ng))) {
// WRF:3632
_result = xg(ng);
// WRF:3633
return _result;
// WRF:3634
}
// WRF:3635
if((x<xg(1))) {
// WRF:3636
_result = 0.0f;
// WRF:3637
return _result;
// WRF:3638
}
// WRF:3639
for(ii=1;ii<=(ng-1);ii+=1) {
// WRF:3640
i = (ng-ii);
// WRF:3641
n = i;
// WRF:3642
np = (n+1);
// WRF:3643
if((x>=xg(i))) {
// WRF:3645
_result = (gamxg(n)+((lake_div(((x-xg(n))),((xg(np)-xg(n)))))*((gamxg(np)-gamxg(n)))));
// WRF:3647
return _result;
// WRF:3648
}
// WRF:3649
}
// WRF:3650
return _result;
return _result;
}
__device__ __forceinline__ float gaml02d300(float x) {
int ig;
int i;
int ii;
int n;
int np;
constexpr int ng=9;
FixedStorage<float,(ng-1+1),1,ng> gamxg{};
FixedStorage<float,(ng-1+1),1,ng> xg{};
float _result;
// WRF:3665
xg.data[0]=0.04f;xg.data[1]=0.075f;xg.data[2]=0.1f;xg.data[3]=0.25f;xg.data[4]=0.5f;xg.data[5]=0.75f;xg.data[6]=1.f;xg.data[7]=2.f;xg.data[8]=10.f;
// WRF:3666
gamxg.data[0]=0.0f;gamxg.data[1]=7.391019203578011e-8f;gamxg.data[2]=0.0002260640810600053f;gamxg.data[3]=0.16567071824457152f;gamxg.data[4]=0.4231369044918005f;gamxg.data[5]=0.5454357914523988f;gamxg.data[6]=0.6170290936864555f;gamxg.data[7]=0.7471346054110058f;gamxg.data[8]=0.9037156157718299f;
// WRF:3673
if((x>=xg(ng))) {
// WRF:3674
_result = xg(ng);
// WRF:3675
return _result;
// WRF:3676
}
// WRF:3677
if((x<xg(1))) {
// WRF:3678
_result = 0.0f;
// WRF:3679
return _result;
// WRF:3680
}
// WRF:3681
for(ii=1;ii<=(ng-1);ii+=1) {
// WRF:3682
i = (ng-ii);
// WRF:3683
n = i;
// WRF:3684
np = (n+1);
// WRF:3685
if((x>=xg(i))) {
// WRF:3687
_result = (gamxg(n)+((lake_div(((x-xg(n))),((xg(np)-xg(n)))))*((gamxg(np)-gamxg(n)))));
// WRF:3689
return _result;
// WRF:3690
}
// WRF:3691
}
// WRF:3692
return _result;
return _result;
}
__device__ __forceinline__ float gaml02d500(float x) {
int ig;
int i;
int ii;
int n;
int np;
constexpr int ng=9;
FixedStorage<float,(ng-1+1),1,ng> gamxg{};
FixedStorage<float,(ng-1+1),1,ng> xg{};
float _result;
// WRF:3711
xg.data[0]=0.04f;xg.data[1]=0.075f;xg.data[2]=0.1f;xg.data[3]=0.25f;xg.data[4]=0.5f;xg.data[5]=0.75f;xg.data[6]=1.f;xg.data[7]=2.f;xg.data[8]=10.f;
// WRF:3712
gamxg.data[0]=0.0f;gamxg.data[1]=0.0f;gamxg.data[2]=2.2346039e-13f;gamxg.data[3]=0.0221272687459f;gamxg.data[4]=0.23556540f;gamxg.data[5]=0.38710348f;gamxg.data[6]=0.48136183f;gamxg.data[7]=0.6565833f;gamxg.data[8]=0.86918315f;
// WRF:3718
if((x>=xg(ng))) {
// WRF:3719
_result = xg(ng);
// WRF:3720
return _result;
// WRF:3721
}
// WRF:3722
if((x<xg(1))) {
// WRF:3723
_result = 0.0f;
// WRF:3724
return _result;
// WRF:3725
}
// WRF:3726
for(ii=1;ii<=(ng-1);ii+=1) {
// WRF:3727
i = (ng-ii);
// WRF:3728
n = i;
// WRF:3729
np = (n+1);
// WRF:3730
if((x>=xg(i))) {
// WRF:3732
_result = (gamxg(n)+((lake_div(((x-xg(n))),((xg(np)-xg(n)))))*((gamxg(np)-gamxg(n)))));
// WRF:3734
return _result;
// WRF:3735
}
// WRF:3736
}
// WRF:3737
return _result;
return _result;
}
__device__ __forceinline__ void setvtz(int ngscnt,Array<const float> qx,Array<const float> qxmin,Array<const float> qxw,Array<float> cx,Array<const float> rho0,Array<const float> rhovt,Array<float> xdia,Array<const float> cno,Array<const float> cnostmp,Array<float> xmas,Array<float> vtxbar,Array<float> xdn,Array<const float> xvmn0,Array<const float> xvmx0,Array<float> xv,Array<const float> cdx,Array<float> cdxgs,int ipconc1,int ndebug1,int ngs,int nz,Array<const int> kgs,Array<const float> fadvisc,float cwmasn,float cwmasx,float cwradn,Array<const float> cnina,float cimna,float cimxa,int itype1a,int itype2a,Array<const float> temcg,int infdo,Array<const float> alpha,int ildo,Array<float> axx,Array<float> bxx) {
int ngs0;
float cwc1;
float fsw;
float swmasmx;
float dtmp;
float cd;
float cwc0;
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwch{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwchl{};
float cwchtmp;
float cwchltmp;
float xnutmp;
float pii;
float cimasx;
float cimasn;
float cwrad;
float vr;
float rnux;
float alp;
float ccimx;
int mgs;
float arx;
float frx;
float vtrain;
float fw;
float fwlo;
float fwhi;
float rfwdiff;
float ar;
float br;
float cs;
float ds;
float gr;
float rwrad;
float rwdia;
float mwfac;
int il;
const float bta1=0.6f;
const float cnit=1.0e-02f;
float x;
float y;
float tmp;
float del;
float aax;
float bbx;
float delrho;
int indxr;
float mwt;
float nwt;
float zwt;
const float rho00=1.225f;
int i;
float xvbarmax;
int l1;
int l2;
// WRF:6312
fwlo = 0.2f;
// WRF:6313
fwhi = 0.4f;
// WRF:6314
rfwdiff = lake_div(1.f,((fwhi-fwlo)));
// WRF:6317
pii = piinv;
// WRF:6319
arx = 10.f;
// WRF:6320
frx = 516.575f;
// WRF:6322
ar = 841.99666f;
// WRF:6323
br = 0.8f;
// WRF:6324
gr = 9.8f;
// WRF:6326
cs = 12.42f;
// WRF:6327
ds = 0.42f;
// WRF:6329
if((ildo==0)) {
// WRF:6330
l1 = lc;
// WRF:6331
l2 = lhab;
// WRF:6332
} else {
// WRF:6333
l1 = ildo;
// WRF:6334
l2 = ildo;
// WRF:6335
}
// WRF:6347
if((lh>1)) {
// WRF:6348
if((dmuh==1.0f)) {
// WRF:6349
cwchtmp = lake_pow((((((3.f+coeff_dnu{}(lh)))*((2.f+coeff_dnu{}(lh))))*((1.0f+coeff_dnu{}(lh))))),(lake_div((-1.f),3.f)));
// WRF:6350
} else {
// WRF:6351
cwchtmp = lake_div(((6.0f*pii)*gamma_sp(lake_div(((coeff_xnu{}(lh)+1.f)),coeff_xmu{}(lh)))),gamma_sp(lake_div(((coeff_xnu{}(lh)+2.f)),coeff_xmu{}(lh))));
// WRF:6352
}
// WRF:6353
}
// WRF:6354
if((lhl>1)) {
// WRF:6355
if((dmuhl==1.0f)) {
// WRF:6356
cwchltmp = lake_pow((((((3.f+coeff_dnu{}(lhl)))*((2.f+coeff_dnu{}(lhl))))*((1.0f+coeff_dnu{}(lhl))))),(lake_div((-1.f),3.f)));
// WRF:6357
} else {
// WRF:6358
cwchltmp = lake_div(((6.0f*pii)*gamma_sp(lake_div(((coeff_xnu{}(lhl)+1)),coeff_xmu{}(lhl)))),gamma_sp(lake_div(((coeff_xnu{}(lhl)+2)),coeff_xmu{}(lhl))));
// WRF:6359
}
// WRF:6360
}
// WRF:6362
if((ipconc<=5)) {
// WRF:6363
if((lh>1)) {for(int _s0=1;_s0<=ngscnt;++_s0) {cwch(_s0) = cwchtmp;}}
// WRF:6364
if((lhl>1)) {for(int _s0=1;_s0<=ngscnt;++_s0) {cwchl(_s0) = cwchltmp;}}
// WRF:6365
} else {
// WRF:6366
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6368
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:6369
if((qx(mgs,lh)>coeff_qxmin{}(lh))) {
// WRF:6370
if((dmuh==1.0f)) {
// WRF:6371
cwch(mgs) = lake_pow((((((3.f+alpha(mgs,lh)))*((2.f+alpha(mgs,lh))))*((1.0f+alpha(mgs,lh))))),(lake_div((-1.f),3.f)));
// WRF:6372
} else {
// WRF:6373
xnutmp = lake_div(((alpha(mgs,lh)-2.0f)),3.0f);
// WRF:6374
cwch(mgs) = lake_div(((6.0f*pii)*gamma_sp(lake_div(((xnutmp+1.f)),coeff_xmu{}(lh)))),gamma_sp(lake_div(((xnutmp+2.f)),coeff_xmu{}(lh))));
// WRF:6375
}
// WRF:6376
} else {
// WRF:6377
cwch(mgs) = cwchtmp;
// WRF:6378
}
// WRF:6379
}
// WRF:6380
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:6381
if((qx(mgs,lhl)>coeff_qxmin{}(lhl))) {
// WRF:6382
if((dmuhl==1.0f)) {
// WRF:6383
cwchl(mgs) = lake_pow((((((3.f+alpha(mgs,lhl)))*((2.f+alpha(mgs,lhl))))*((1.0f+alpha(mgs,lhl))))),(lake_div((-1.f),3.f)));
// WRF:6384
} else {
// WRF:6385
xnutmp = lake_div(((alpha(mgs,lhl)-2.0f)),3.0f);
// WRF:6386
cwchl(mgs) = lake_div(((6.0f*pii)*gamma_sp(lake_div(((xnutmp+1)),coeff_xmu{}(lhl)))),gamma_sp(lake_div(((xnutmp+2)),coeff_xmu{}(lhl))));
// WRF:6387
}
// WRF:6388
} else {
// WRF:6389
cwchl(mgs) = cwchltmp;
// WRF:6390
}
// WRF:6391
}
// WRF:6393
}
// WRF:6395
}
// WRF:6398
cimasn = lake_min(cimas0,6.88e-13f);
// WRF:6399
cimasx = 1.0e-8f;
// WRF:6400
ccimx = 5000.0e3f;
// WRF:6402
cwc1 = lake_div(6.0f,((pi*1000.f)));
// WRF:6403
cwc0 = pii;
// WRF:6404
mwfac = lake_pow(6.0f,(lake_div(1.f,3.f)));
// WRF:6407
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6418
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6420
if(((ildo==0)||(ildo==lc))) {
// WRF:6422
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6423
xv(mgs,lc) = 0.0f;
// WRF:6425
if((qx(mgs,lc)>coeff_qxmin{}(lc))) {
// WRF:6427
if((ipconc>=2)) {
// WRF:6428
if((cx(mgs,lc)>cxmin)) {
// WRF:6429
xmas(mgs,lc) = lake_min(lake_max(lake_div((qx(mgs,lc)*rho0(mgs)),cx(mgs,lc)),cwmasn),cwmasx);
// WRF:6431
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6432
} else {
// WRF:6433
cx(mgs,lc) = lake_max(cxmin,lake_div((rho0(mgs)*qx(mgs,lc)),cwmasx));
// WRF:6434
xmas(mgs,lc) = lake_min(lake_max(lake_div((qx(mgs,lc)*rho0(mgs)),cx(mgs,lc)),cwmasn),cwmasx);
// WRF:6435
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6437
}
// WRF:6438
} else {
// WRF:6439
if((ipconc<2)) {
// WRF:6440
cx(mgs,lc) = lake_div((rho0(mgs)*ccn),rho00);
// WRF:6441
}
// WRF:6442
if(((qx(mgs,lc)>coeff_qxmin{}(lc))&&(cx(mgs,lc)>0.01f))) {
// WRF:6443
xmas(mgs,lc) = lake_min(lake_max(lake_div((qx(mgs,lc)*rho0(mgs)),cx(mgs,lc)),(xdn(mgs,lc)*coeff_xvmn{}(lc))),(xdn(mgs,lc)*coeff_xvmx{}(lc)));
// WRF:6447
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6448
cx(mgs,lc) = lake_div((qx(mgs,lc)*rho0(mgs)),xmas(mgs,lc));
// WRF:6450
} else if(((qx(mgs,lc)>coeff_qxmin{}(lc))&&(cx(mgs,lc)<=1.0e-9f))) {
// WRF:6451
cx(mgs,lc) = lake_max(cxmin,lake_div((rho0(mgs)*qx(mgs,lc)),cwmasx));
// WRF:6452
xmas(mgs,lc) = lake_min(lake_max(lake_div((qx(mgs,lc)*rho0(mgs)),cx(mgs,lc)),cwmasn),cwmasx);
// WRF:6454
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6456
} else if(((qx(mgs,lc)>coeff_qxmin{}(lc))&&(cx(mgs,lc)<=0.01f))) {
// WRF:6457
xmas(mgs,lc) = (lake_div(((xdn(mgs,lc)*4.f)*pi),3.f)*lake_pow((5.0e-6f),3));
// WRF:6458
cx(mgs,lc) = lake_div((rho0(mgs)*qx(mgs,lc)),xmas(mgs,lc));
// WRF:6459
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6461
} else {
// WRF:6462
xmas(mgs,lc) = cwmasn;
// WRF:6463
xv(mgs,lc) = lake_div(xmas(mgs,lc),1000.f);
// WRF:6465
}
// WRF:6466
}
// WRF:6475
xdia(mgs,lc,1) = lake_pow(((xmas(mgs,lc)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6476
xdia(mgs,lc,2) = lake_pow(xdia(mgs,lc,1),2);
// WRF:6477
xdia(mgs,lc,3) = xdia(mgs,lc,1);
// WRF:6478
cwrad = (0.5f*xdia(mgs,lc,1));
// WRF:6479
if((fadvisc(mgs)>0.0f)) {
// WRF:6480
vtxbar(mgs,lc,1) = lake_div(((((2.0f*gr)*xdn(mgs,lc))*(lake_pow(cwrad,2)))),((9.0f*fadvisc(mgs))));
// WRF:6483
} else {
// WRF:6484
vtxbar(mgs,lc,1) = 0.0f;
// WRF:6485
}
// WRF:6488
} else {
// WRF:6489
xmas(mgs,lc) = cwmasn;
// WRF:6490
xv(mgs,lc) = lake_div(xmas(mgs,lc),xdn(mgs,lc));
// WRF:6491
if((qx(mgs,lc)<=0.0f)) {cx(mgs,lc) = 0.0f;}
// WRF:6492
if((ipconc<=1)) {cx(mgs,lc) = 0.01f;}
// WRF:6493
xdia(mgs,lc,1) = (2.f*cwradn);
// WRF:6494
xdia(mgs,lc,2) = (4.f*lake_pow(cwradn,2));
// WRF:6495
xdia(mgs,lc,3) = xdia(mgs,lc,1);
// WRF:6496
vtxbar(mgs,lc,1) = 0.0f;
// WRF:6498
}
// WRF:6500
}
// WRF:6502
}
// WRF:6514
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6516
if(((li>1)&&(((ildo==0)||(ildo==li))))) {
// WRF:6517
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6518
xdn(mgs,li) = 900.0f;
// WRF:6519
if((ipconc==0)) {
// WRF:6521
cx(mgs,li) = cnina(mgs);
// WRF:6522
if((cimna>1.0f)) {
// WRF:6523
cx(mgs,li) = lake_max(cimna,cx(mgs,li));
// WRF:6524
}
// WRF:6525
if((cimxa>1.0f)) {
// WRF:6526
cx(mgs,li) = lake_min(cimxa,cx(mgs,li));
// WRF:6527
}
// WRF:6529
if(((itype1a>=1)||(itype2a>=1))) {
// WRF:6530
cx(mgs,li) = lake_max(cx(mgs,li),lake_div((qx(mgs,li)*rho0(mgs)),cimasx));
// WRF:6531
cx(mgs,li) = lake_min(cx(mgs,li),lake_div((qx(mgs,li)*rho0(mgs)),cimasn));
// WRF:6532
}
// WRF:6534
cx(mgs,li) = lake_max(1.0e-20f,cx(mgs,li));
// WRF:6538
} else if((ipconc>=1)) {
// WRF:6539
if((qx(mgs,li)>coeff_qxmin{}(li))) {
// WRF:6540
cx(mgs,li) = lake_max(cx(mgs,li),lake_div((qx(mgs,li)*rho0(mgs)),cimasx));
// WRF:6541
cx(mgs,li) = lake_min(cx(mgs,li),lake_div((qx(mgs,li)*rho0(mgs)),cimasn));
// WRF:6543
}
// WRF:6544
}
// WRF:6546
if((qx(mgs,li)>coeff_qxmin{}(li))) {
// WRF:6547
xmas(mgs,li) = lake_max(lake_div((qx(mgs,li)*rho0(mgs)),cx(mgs,li)),cimasn);
// WRF:6554
if((xmas(mgs,li)>0.0f)) {
// WRF:6559
if((ixtaltype==1)) {
// WRF:6560
xdia(mgs,li,1) = (0.1871f*(lake_pow(xmas(mgs,li),(0.3429f))));
// WRF:6561
xdia(mgs,li,3) = (0.1871f*(lake_pow(xmas(mgs,li),(0.3429f))));
// WRF:6562
} else if((ixtaltype==2)) {
// WRF:6563
xdia(mgs,li,1) = (0.277823f*lake_pow(xmas(mgs,li),0.359971f));
// WRF:6564
xdia(mgs,li,3) = (0.277823f*lake_pow(xmas(mgs,li),0.359971f));
// WRF:6565
}
// WRF:6566
}
// WRF:6571
if((ipconc>=0)) {
// WRF:6574
xv(mgs,li) = lake_div(xmas(mgs,li),xdn(mgs,li));
// WRF:6575
if((icefallopt==1)) {
// WRF:6576
if((ixtaltype==1)) {
// WRF:6577
tmp = lake_div(((67056.6300748612f*rhovt(mgs))),((lake_pow((lake_div(((1.0f+cinu)),xv(mgs,li))),0.4716666666666667f)*gfcinu1)));
// WRF:6579
vtxbar(mgs,li,2) = (tmp*gfcinu1p47);
// WRF:6580
vtxbar(mgs,li,1) = lake_div((tmp*gfcinu2p47),((1.f+cinu)));
// WRF:6581
vtxbar(mgs,li,3) = vtxbar(mgs,li,1);
// WRF:6582
} else if((ixtaltype==2)) {
// WRF:6583
vtxbar(mgs,li,1) = ((11.9495f*rhovt(mgs))*lake_pow((xv(mgs,li)),(0.14f)));
// WRF:6584
vtxbar(mgs,li,2) = ((7.02909f*rhovt(mgs))*lake_pow((xv(mgs,li)),(0.14f)));
// WRF:6585
vtxbar(mgs,li,3) = vtxbar(mgs,li,1);
// WRF:6587
}
// WRF:6589
} else if((icefallopt==2)) {
// WRF:6590
tmp = lake_div(((82.3166f*rhovt(mgs))),((lake_pow((lake_div(((1.0f+cinu)),xv(mgs,li))),0.22117f)*gfcinu1)));
// WRF:6592
vtxbar(mgs,li,2) = (tmp*gfcinu1p22);
// WRF:6593
vtxbar(mgs,li,1) = lake_div((tmp*gfcinu2p22),((1.f+cinu)));
// WRF:6594
vtxbar(mgs,li,3) = vtxbar(mgs,li,1);
// WRF:6596
} else if((icefallopt==3)) {
// WRF:6598
tmp = lake_div(((47.6273f*rhovt(mgs))),((lake_pow((lake_div(((1.0f+cinu)),xv(mgs,li))),0.18333f)*gfcinu1)));
// WRF:6600
vtxbar(mgs,li,2) = (tmp*gfcinu1p18);
// WRF:6601
vtxbar(mgs,li,1) = lake_div((tmp*gfcinu2p18),((1.f+cinu)));
// WRF:6602
vtxbar(mgs,li,3) = vtxbar(mgs,li,1);
// WRF:6604
}
// WRF:6608
xdia(mgs,li,2) = lake_pow(xdia(mgs,li,1),2);
// WRF:6610
} else {
// WRF:6611
xdia(mgs,li,1) = lake_max(xdia(mgs,li,1),10.e-6f);
// WRF:6612
xdia(mgs,li,1) = lake_min(xdia(mgs,li,1),1000.e-6f);
// WRF:6613
vtxbar(mgs,li,1) = ((4.942e4f)*(lake_pow(xdia(mgs,li,1),(1.4150f))));
// WRF:6615
xdn(mgs,li) = 900.0f;
// WRF:6616
xdia(mgs,li,2) = lake_pow(xdia(mgs,li,1),2);
// WRF:6617
vtxbar(mgs,li,1) = (vtxbar(mgs,li,1)*rhovt(mgs));
// WRF:6618
xv(mgs,li) = lake_div(xmas(mgs,li),xdn(mgs,li));
// WRF:6619
}
// WRF:6620
} else {
// WRF:6621
xmas(mgs,li) = 1.e-13f;
// WRF:6622
if((qx(mgs,li)<=0.0f)) {cx(mgs,li) = 0.0f;}
// WRF:6623
xdn(mgs,li) = 900.0f;
// WRF:6624
xdia(mgs,li,1) = 1.e-7f;
// WRF:6625
xdia(mgs,li,2) = (1.e-14f);
// WRF:6626
xdia(mgs,li,3) = 1.e-7f;
// WRF:6627
vtxbar(mgs,li,1) = 0.0f;
// WRF:6630
}
// WRF:6632
if((icefallfac!=1.0f)) {
// WRF:6633
vtxbar(mgs,li,1) = (icefallfac*vtxbar(mgs,li,1));
// WRF:6634
vtxbar(mgs,li,2) = (icefallfac*vtxbar(mgs,li,2));
// WRF:6635
vtxbar(mgs,li,3) = (icefallfac*vtxbar(mgs,li,3));
// WRF:6636
}
// WRF:6640
}
// WRF:6642
}
// WRF:6651
if(((ildo==0)||(ildo==lr))) {
// WRF:6652
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6653
if((qx(mgs,lr)>coeff_qxmin{}(lr))) {
// WRF:6658
if((ipconc>=3)) {
// WRF:6659
xv(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((xdn(mgs,lr)*lake_max(1.0e-11f,cx(mgs,lr)))));
// WRF:6660
xvbarmax = coeff_xvmx{}(lr);
// WRF:6661
if((imaxdiaopt==1)) {
// WRF:6662
xvbarmax = coeff_xvmx{}(lr);
// WRF:6663
} else if((imaxdiaopt==2)) {
// WRF:6664
if((imurain==1)) {
// WRF:6665
xvbarmax = lake_div(coeff_xvmx{}(lr),(lake_div(lake_pow(((3.f+alpha(mgs,lr))),3),(((((3.f+alpha(mgs,lr)))*((2.f+alpha(mgs,lr))))*((1.f+alpha(mgs,lr))))))));
// WRF:6666
} else if((imurain==3)) {
// WRF:6668
}
// WRF:6669
} else if((imaxdiaopt==3)) {
// WRF:6670
if((imurain==1)) {
// WRF:6671
xvbarmax = lake_div(coeff_xvmx{}(lr),(lake_div(lake_pow(((4.f+alpha(mgs,lr))),3),(((((3.f+alpha(mgs,lr)))*((2.f+alpha(mgs,lr))))*((1.f+alpha(mgs,lr))))))));
// WRF:6672
} else if((imurain==3)) {
// WRF:6674
}
// WRF:6675
}
// WRF:6677
if((xv(mgs,lr)>xvbarmax)) {
// WRF:6678
xv(mgs,lr) = xvbarmax;
// WRF:6679
cx(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((xvbarmax*xdn(mgs,lr))));
// WRF:6680
} else if((xv(mgs,lr)<coeff_xvmn{}(lr))) {
// WRF:6681
xv(mgs,lr) = coeff_xvmn{}(lr);
// WRF:6682
cx(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((coeff_xvmn{}(lr)*xdn(mgs,lr))));
// WRF:6683
}
// WRF:6686
xmas(mgs,lr) = (xv(mgs,lr)*xdn(mgs,lr));
// WRF:6687
xdia(mgs,lr,3) = lake_pow(((xmas(mgs,lr)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6688
if((imurain==3)) {
// WRF:6690
xdia(mgs,lr,1) = xdia(mgs,lr,3);
// WRF:6691
} else {
// WRF:6692
xdia(mgs,lr,1) = lake_pow((lake_div(((6.f*pii)*xv(mgs,lr)),(((((alpha(mgs,lr)+3.f))*((alpha(mgs,lr)+2.f)))*((alpha(mgs,lr)+1.f)))))),(lake_div(1.f,3.f)));
// WRF:6693
}
// WRF:6700
} else {
// WRF:6701
xdia(mgs,lr,1) = lake_pow((lake_div((qx(mgs,lr)*rho0(mgs)),(((pi*xdn(mgs,lr))*coeff_cno{}(lr))))),(0.25f));
// WRF:6703
xmas(mgs,lr) = ((xdn(mgs,lr)*(lake_div(pi,6.f)))*lake_pow(xdia(mgs,lr,1),3));
// WRF:6704
xdia(mgs,lr,3) = lake_pow(((xmas(mgs,lr)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6705
cx(mgs,lr) = (coeff_cno{}(lr)*xdia(mgs,lr,1));
// WRF:6706
xv(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((xdn(mgs,lr)*cx(mgs,lr))));
// WRF:6707
}
// WRF:6708
} else {
// WRF:6709
xdia(mgs,lr,1) = 1.e-9f;
// WRF:6710
xdia(mgs,lr,3) = 1.e-9f;
// WRF:6711
xmas(mgs,lr) = ((xdn(mgs,lr)*(lake_div(pi,6.f)))*lake_pow(xdia(mgs,lr,1),3));
// WRF:6713
}
// WRF:6714
xdia(mgs,lr,2) = lake_pow(xdia(mgs,lr,1),2);
// WRF:6716
}
// WRF:6718
}
// WRF:6724
if(((ls>1)&&(((ildo==0)||(ildo==ls))))) {
// WRF:6726
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6727
if((qx(mgs,ls)>coeff_qxmin{}(ls))) {
// WRF:6728
if((ipconc>=4)) {
// WRF:6730
xmas(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),(lake_max(1.0e-9f,cx(mgs,ls))));
// WRF:6731
swmasmx = 13.7e-6f;
// WRF:6737
if((isnowdens==2)) {
// WRF:6739
xdn(mgs,ls) = (0.0346159f*sqrtf(lake_div(cx(mgs,ls),((qx(mgs,ls)*rho0(mgs))))));
// WRF:6740
xdn(mgs,ls) = lake_max(100.0f,xdn(mgs,ls));
// WRF:6742
if((xdn(mgs,ls)<=900.f)) {
// WRF:6743
dtmp = sqrtf(lake_div(xmas(mgs,ls),0.069f));
// WRF:6744
xv(mgs,ls) = (28.8887f*lake_pow(xmas(mgs,ls),(lake_div(3.f,2.f))));
// WRF:6745
} else {
// WRF:6746
xdn(mgs,ls) = 900.f;
// WRF:6747
xv(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),((xdn(mgs,ls)*lake_max(1.0e-9f,cx(mgs,ls)))));
// WRF:6748
dtmp = lake_pow((((xv(mgs,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6749
}
// WRF:6751
} else {
// WRF:6752
xv(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),((xdn(mgs,ls)*lake_max(1.0e-9f,cx(mgs,ls)))));
// WRF:6753
dtmp = lake_pow((((xv(mgs,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6754
}
// WRF:6756
xdia(mgs,ls,1) = dtmp;
// WRF:6758
if(((xv(mgs,ls)<coeff_xvmn{}(ls))&&(isnowdens==1))) {
// WRF:6759
xv(mgs,ls) = lake_max(coeff_xvmn{}(ls),xv(mgs,ls));
// WRF:6760
xmas(mgs,ls) = (xv(mgs,ls)*xdn(mgs,ls));
// WRF:6761
cx(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),(xmas(mgs,ls)));
// WRF:6762
xdia(mgs,ls,1) = lake_pow((((xv(mgs,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6763
}
// WRF:6765
if((xv(mgs,ls)>(coeff_xvmx{}(ls)*lake_max(1.f,lake_div(100.f,lake_min(100.f,xdn(mgs,ls))))))) {
// WRF:6766
xv(mgs,ls) = lake_min(coeff_xvmx{}(ls),lake_max(coeff_xvmn{}(ls),xv(mgs,ls)));
// WRF:6767
xmas(mgs,ls) = (0.106214f*lake_pow(xv(mgs,ls),(lake_div(2.f,3.f))));
// WRF:6768
cx(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),(xmas(mgs,ls)));
// WRF:6769
xdn(mgs,ls) = (0.0346159f*sqrtf(lake_div(cx(mgs,ls),((qx(mgs,ls)*rho0(mgs))))));
// WRF:6770
xdia(mgs,ls,1) = sqrtf(lake_div(xmas(mgs,ls),0.069f));
// WRF:6771
}
// WRF:6773
xdia(mgs,ls,3) = xdia(mgs,ls,1);
// WRF:6775
} else {
// WRF:6776
xdia(mgs,ls,1) = lake_pow((lake_div((qx(mgs,ls)*rho0(mgs)),(((pi*xdn(mgs,ls))*cnostmp(mgs))))),(0.25f));
// WRF:6778
cx(mgs,ls) = (cnostmp(mgs)*xdia(mgs,ls,1));
// WRF:6779
xv(mgs,ls) = lake_div((rho0(mgs)*qx(mgs,ls)),((xdn(mgs,ls)*cx(mgs,ls))));
// WRF:6780
xdia(mgs,ls,3) = lake_pow((((xv(mgs,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6781
}
// WRF:6782
} else {
// WRF:6783
xdia(mgs,ls,1) = 1.e-9f;
// WRF:6784
xdia(mgs,ls,3) = 1.e-9f;
// WRF:6785
cx(mgs,ls) = 0.0f;
// WRF:6787
if((isnowdens==2)) {
// WRF:6788
xdn(mgs,ls) = 90.f;
// WRF:6789
}
// WRF:6791
}
// WRF:6792
xdia(mgs,ls,2) = lake_pow(xdia(mgs,ls,1),2);
// WRF:6795
}
// WRF:6797
}
// WRF:6805
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:6807
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6808
if((qx(mgs,lh)>coeff_qxmin{}(lh))) {
// WRF:6809
if((ipconc>=5)) {
// WRF:6811
xv(mgs,lh) = lake_div((rho0(mgs)*qx(mgs,lh)),((xdn(mgs,lh)*lake_max(1.0e-9f,cx(mgs,lh)))));
// WRF:6812
xmas(mgs,lh) = (xv(mgs,lh)*xdn(mgs,lh));
// WRF:6814
if(((xv(mgs,lh)<coeff_xvmn{}(lh))||(xv(mgs,lh)>coeff_xvmx{}(lh)))) {
// WRF:6815
xv(mgs,lh) = lake_min(coeff_xvmx{}(lh),lake_max(coeff_xvmn{}(lh),xv(mgs,lh)));
// WRF:6816
xmas(mgs,lh) = (xv(mgs,lh)*xdn(mgs,lh));
// WRF:6817
cx(mgs,lh) = lake_div((rho0(mgs)*qx(mgs,lh)),(xmas(mgs,lh)));
// WRF:6818
}
// WRF:6820
xdia(mgs,lh,3) = lake_pow((((xv(mgs,lh)*6.f)*pii)),(lake_div(1.f,3.f)));
// WRF:6821
if((dmuh==1.0f)) {
// WRF:6822
xdia(mgs,lh,1) = (cwch(mgs)*xdia(mgs,lh,3));
// WRF:6823
} else {
// WRF:6824
xdia(mgs,lh,1) = lake_pow(((xv(mgs,lh)*cwch(mgs))),(lake_div(1.f,3.f)));
// WRF:6825
}
// WRF:6827
} else {
// WRF:6828
xdia(mgs,lh,1) = lake_pow((lake_div((qx(mgs,lh)*rho0(mgs)),(((pi*xdn(mgs,lh))*coeff_cno{}(lh))))),(0.25f));
// WRF:6830
cx(mgs,lh) = (coeff_cno{}(lh)*xdia(mgs,lh,1));
// WRF:6831
xv(mgs,lh) = lake_max(coeff_xvmn{}(lh),lake_div((rho0(mgs)*qx(mgs,lh)),((xdn(mgs,lh)*cx(mgs,lh)))));
// WRF:6832
xdia(mgs,lh,3) = lake_pow((lake_div((xv(mgs,lh)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6833
}
// WRF:6834
} else {
// WRF:6835
xdia(mgs,lh,1) = 1.e-9f;
// WRF:6836
xdia(mgs,lh,3) = 1.e-9f;
// WRF:6837
}
// WRF:6838
xdia(mgs,lh,2) = lake_pow(xdia(mgs,lh,1),2);
// WRF:6841
}
// WRF:6843
}
// WRF:6851
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:6853
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6854
if((qx(mgs,lhl)>coeff_qxmin{}(lhl))) {
// WRF:6855
if((ipconc>=5)) {
// WRF:6857
xv(mgs,lhl) = lake_div((rho0(mgs)*qx(mgs,lhl)),((xdn(mgs,lhl)*lake_max(1.0e-9f,cx(mgs,lhl)))));
// WRF:6858
xmas(mgs,lhl) = (xv(mgs,lhl)*xdn(mgs,lhl));
// WRF:6861
if(((xv(mgs,lhl)<coeff_xvmn{}(lhl))||(xv(mgs,lhl)>coeff_xvmx{}(lhl)))) {
// WRF:6862
xv(mgs,lhl) = lake_min(coeff_xvmx{}(lhl),lake_max(coeff_xvmn{}(lhl),xv(mgs,lhl)));
// WRF:6863
xmas(mgs,lhl) = (xv(mgs,lhl)*xdn(mgs,lhl));
// WRF:6864
cx(mgs,lhl) = lake_div((rho0(mgs)*qx(mgs,lhl)),(xmas(mgs,lhl)));
// WRF:6865
}
// WRF:6867
xdia(mgs,lhl,3) = lake_pow((lake_div((xv(mgs,lhl)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6868
if((dmuhl==1.0f)) {
// WRF:6869
xdia(mgs,lhl,1) = (cwchl(mgs)*xdia(mgs,lhl,3));
// WRF:6870
} else {
// WRF:6871
xdia(mgs,lhl,1) = lake_pow(((xv(mgs,lhl)*cwchl(mgs))),(lake_div(1.f,3.f)));
// WRF:6872
}
// WRF:6875
} else {
// WRF:6876
xdia(mgs,lhl,1) = lake_pow((lake_div((qx(mgs,lhl)*rho0(mgs)),(((pi*xdn(mgs,lhl))*coeff_cno{}(lhl))))),(0.25f));
// WRF:6878
cx(mgs,lhl) = (coeff_cno{}(lhl)*xdia(mgs,lhl,1));
// WRF:6879
xv(mgs,lhl) = lake_max(coeff_xvmn{}(lhl),lake_div((rho0(mgs)*qx(mgs,lhl)),((xdn(mgs,lhl)*cx(mgs,lhl)))));
// WRF:6880
xdia(mgs,lhl,3) = lake_pow((lake_div((xv(mgs,lhl)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6881
}
// WRF:6882
} else {
// WRF:6883
xdia(mgs,lhl,1) = 1.e-9f;
// WRF:6884
xdia(mgs,lhl,3) = 1.e-9f;
// WRF:6885
}
// WRF:6886
xdia(mgs,lhl,2) = lake_pow(xdia(mgs,lhl,1),2);
// WRF:6889
}
// WRF:6891
}
// WRF:6907
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6914
if(((ildo==0)||(ildo==lr))) {
// WRF:6915
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:6916
if((qx(mgs,lr)>coeff_qxmin{}(lr))) {
// WRF:6917
if((ipconc<3)) {
// WRF:6918
vtxbar(mgs,lr,1) = (((rainfallfac*(lake_div((ar*gf4br),6.0f)))*(lake_pow(xdia(mgs,lr,1),br)))*rhovt(mgs));
// WRF:6920
} else {
// WRF:6922
if((imurain==1)) {
// WRF:6929
alp = alpha(mgs,lr);
// WRF:6931
vtxbar(mgs,lr,1) = ((rhovt(mgs)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(mgs,lr,1)))),(((-alp)-4.0f))))));
// WRF:6933
if(((infdo>=1)&&(rssflg==1))) {
// WRF:6934
vtxbar(mgs,lr,2) = ((rhovt(mgs)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(mgs,lr,1)))),(((-alp)-1.0f))))));
// WRF:6935
} else {
// WRF:6936
vtxbar(mgs,lr,2) = vtxbar(mgs,lr,1);
// WRF:6937
}
// WRF:6939
if(((infdo>=2)&&(rssflg==1))) {
// WRF:6940
vtxbar(mgs,lr,3) = ((rhovt(mgs)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(mgs,lr,1)))),(((-alp)-7.0f))))));
// WRF:6941
} else {
// WRF:6942
vtxbar(mgs,lr,3) = vtxbar(mgs,lr,1);
// WRF:6943
}
// WRF:6947
} else if((imurain==3)) {
// WRF:6949
if((lzr<1)) {
// WRF:6950
rwdia = lake_min(xdia(mgs,lr,1),8.0e-3f);
// WRF:6952
vtxbar(mgs,lr,1) = (((rhovt(mgs)*6.0f)*pii)*(((((0.04771f+(3788.0f*rwdia))-(1.105e6f*lake_pow(rwdia,2)))+(1.412e8f*lake_pow(rwdia,3)))-(6.527e9f*lake_pow(rwdia,4)))));
// WRF:6955
if((infdo>=1)) {
// WRF:6956
if((rssflg>=1)) {
// WRF:6957
vtxbar(mgs,lr,2) = ((((((0.09112f+(2714.0f*rwdia))-(4.872e5f*lake_pow(rwdia,2)))+(4.495e7f*lake_pow(rwdia,3)))-(1.626e9f*lake_pow(rwdia,4))))*rhovt(mgs));
// WRF:6959
} else {
// WRF:6960
vtxbar(mgs,lr,2) = vtxbar(mgs,lr,1);
// WRF:6961
}
// WRF:6962
}
// WRF:6964
if((infdo>=2)) {
// WRF:6965
vtxbar(mgs,lr,3) = (rhovt(mgs)*(((((0.0911229f+(9246.494f*(rwdia)))-(3.2839926e6f*(lake_pow(rwdia,2))))+(4.944093e8f*(lake_pow(rwdia,3))))-(2.631718e10f*(lake_pow(rwdia,4))))));
// WRF:6971
}
// WRF:6973
} else {
// WRF:6975
vr = xv(mgs,lr);
// WRF:6976
rnux = alpha(mgs,lr);
// WRF:6978
if(((infdo>=1)&&(rssflg==1))) {
// WRF:6979
vtxbar(mgs,lr,2) = lake_div((rhovt(mgs)*((lake_pow((lake_div(((1.f+rnux)),vr)),((-1.333333f)))*(((((((0.0911229f*lake_pow((lake_div(((1.f+rnux)),vr)),1.333333f))*gamma_sp((1.f+rnux)))+lake_div((((5430.3131f*((1.f+rnux)))*gamma_sp((lake_div(4.f,3.f)+rnux)))),vr))-((1.0732802e6f*lake_pow((lake_div(((1.f+rnux)),vr)),0.6666667f))*gamma_sp((1.666667f+rnux))))+((8.584110982429507e7f*lake_pow((lake_div(((1.f+rnux)),vr)),(lake_div(1.f,3.f))))*gamma_sp((2.f+rnux))))-(2.3303765697228556e9f*gamma_sp((lake_div(7.f,3.f)+rnux)))))))),gamma_sp((1.f+rnux)));
// WRF:6989
}
// WRF:6992
vtxbar(mgs,lr,1) = lake_div((rhovt(mgs)*(((((((0.0911229f*lake_pow(((1+rnux)),1.3333333333333333f))*gamma_sp((2.f+rnux)))+(((5430.313059683277f*((1+rnux)))*lake_pow(vr,0.3333333333333333f))*gamma_sp((2.333333333333333f+rnux))))-(((1.0732802065650471e6f*lake_pow(((1+rnux)),0.6666666666666666f))*lake_pow(vr,0.6666666666666666f))*gamma_sp((2.6666666666666667f+rnux))))+(((8.584110982429507e7f*lake_pow(((1+rnux)),0.3333333333333333f))*vr)*gamma_sp((3+rnux))))-((2.3303765697228556e9f*lake_pow(vr,1.3333333333333333f))*gamma_sp((3.333333333333333f+rnux)))))),((lake_pow(((1+rnux)),2.333333333333333f)*gamma_sp((1+rnux)))));
// WRF:7003
if(((infdo>=1)&&(rssflg==0))) {
// WRF:7004
vtxbar(mgs,lr,2) = vtxbar(mgs,lr,1);
// WRF:7005
}
// WRF:7007
if(((infdo>=2)&&(rssflg==1))) {
// WRF:7008
vtxbar(mgs,lr,3) = lake_div((rhovt(mgs)*((((1.f+rnux))*(((((((0.0911229f*lake_pow(((1+rnux)),1.3333333333333333f))*gamma_sp((3.f+rnux)))+(((5430.313059683277f*((1+rnux)))*lake_pow(vr,0.3333333333333333f))*gamma_sp((3.3333333333333335f+rnux))))-(((1.0732802065650471e6f*lake_pow(((1+rnux)),0.6666666666666666f))*lake_pow(vr,0.6666666666666666f))*gamma_sp((3.6666666666666665f+rnux))))+(((8.5841109824295e7f*lake_pow(((1+rnux)),0.3333333333333333f))*vr)*gamma_sp((4.f+rnux))))-((2.3303765697228556e9f*lake_pow(vr,1.3333333333333333f))*gamma_sp((4.333333333333333f+rnux)))))))),(((lake_pow(((1+rnux)),3.3333333333333335f)*((2+rnux)))*gamma_sp((1+rnux)))));
// WRF:7022
} else if((infdo>=2)) {
// WRF:7023
vtxbar(mgs,lr,3) = vtxbar(mgs,lr,1);
// WRF:7024
}
// WRF:7027
}
// WRF:7028
}
// WRF:7040
}
// WRF:7041
} else {
// WRF:7042
vtxbar(mgs,lr,1) = 0.0f;
// WRF:7043
vtxbar(mgs,lr,2) = 0.0f;
// WRF:7044
}
// WRF:7045
}
// WRF:7046
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7048
}
// WRF:7054
if(((ls>1)&&(((ildo==0)||(ildo==ls))))) {
// WRF:7055
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7056
if((qx(mgs,ls)>coeff_qxmin{}(ls))) {
// WRF:7057
if((ipconc>=4)) {
// WRF:7058
if((mixedphase&&qsvtmod)) {
// WRF:7059
} else {
// WRF:7060
if((isnowfall==1)) {
// WRF:7062
vtxbar(mgs,ls,1) = ((5.72462f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(lake_div(1.f,12.f))));
// WRF:7063
} else if((isnowfall==2)) {
// WRF:7065
if((isnowdens==1)) {
// WRF:7066
vtxbar(mgs,ls,1) = ((11.9495f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(0.14f)));
// WRF:7067
} else {
// WRF:7068
vtxbar(mgs,ls,1) = ((11.9495f*rhovt(mgs))*lake_pow((lake_div((xv(mgs,ls)*xdn(mgs,ls)),100.f)),(0.14f)));
// WRF:7069
}
// WRF:7070
} else if((isnowfall==3)) {
// WRF:7072
vtxbar(mgs,ls,1) = ((50.092f*rhovt(mgs))*lake_pow((xmas(mgs,ls)),(0.2635f)));
// WRF:7073
}
// WRF:7075
if((lake_abs(sssflg)>=1)) {
// WRF:7076
if((isnowfall==1)) {
// WRF:7077
vtxbar(mgs,ls,2) = ((4.04091f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(lake_div(1.f,12.f))));
// WRF:7078
} else if((isnowfall==2)) {
// WRF:7080
if((isnowdens==1)) {
// WRF:7081
vtxbar(mgs,ls,2) = ((7.02909f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(0.14f)));
// WRF:7082
} else {
// WRF:7083
vtxbar(mgs,ls,2) = ((7.02909f*rhovt(mgs))*lake_pow((lake_div((xv(mgs,ls)*xdn(mgs,ls)),100.f)),(0.14f)));
// WRF:7084
}
// WRF:7085
} else if((isnowfall==3)) {
// WRF:7087
vtxbar(mgs,ls,2) = ((21.6147f*rhovt(mgs))*lake_pow((xmas(mgs,ls)),(0.2635f)));
// WRF:7088
}
// WRF:7089
} else {
// WRF:7090
vtxbar(mgs,ls,2) = vtxbar(mgs,ls,1);
// WRF:7091
}
// WRF:7092
if((infdo>=2)) {
// WRF:7093
if((isnowfall==1)) {
// WRF:7094
vtxbar(mgs,ls,3) = ((6.12217f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(lake_div(1.f,12.f))));
// WRF:7095
} else if((isnowfall==2)) {
// WRF:7096
vtxbar(mgs,ls,3) = ((13.3436f*rhovt(mgs))*lake_pow((xv(mgs,ls)),(0.14f)));
// WRF:7097
} else if((isnowfall==3)) {
// WRF:7099
vtxbar(mgs,ls,3) = ((61.0914f*rhovt(mgs))*lake_pow((xmas(mgs,ls)),(0.2635f)));
// WRF:7100
}
// WRF:7101
}
// WRF:7103
if(((sssflg<0)&&(temcg(mgs)>lake_abs(sssflg)))) {
// WRF:7104
vtxbar(mgs,ls,2) = vtxbar(mgs,ls,1);
// WRF:7105
vtxbar(mgs,ls,3) = vtxbar(mgs,ls,1);
// WRF:7106
}
// WRF:7108
}
// WRF:7109
} else {
// WRF:7110
vtxbar(mgs,ls,1) = (((lake_div((cs*gf4ds),6.0f))*(lake_pow(xdia(mgs,ls,1),ds)))*rhovt(mgs));
// WRF:7111
vtxbar(mgs,ls,2) = vtxbar(mgs,ls,1);
// WRF:7112
}
// WRF:7113
} else {
// WRF:7114
vtxbar(mgs,ls,1) = 0.0f;
// WRF:7115
}
// WRF:7117
if((snowfallfac!=1.0f)) {
// WRF:7118
vtxbar(mgs,ls,1) = (snowfallfac*vtxbar(mgs,ls,1));
// WRF:7119
vtxbar(mgs,ls,2) = (snowfallfac*vtxbar(mgs,ls,2));
// WRF:7120
vtxbar(mgs,ls,3) = (snowfallfac*vtxbar(mgs,ls,3));
// WRF:7121
}
// WRF:7124
}
// WRF:7125
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7127
}
// WRF:7134
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:7136
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7137
vtxbar(mgs,lh,1) = 0.0f;
// WRF:7138
if((qx(mgs,lh)>coeff_qxmin{}(lh))) {
// WRF:7139
cd = coeff_cdx{}(lh);
// WRF:7140
if((icdx==1)) {
// WRF:7141
cd = coeff_cdx{}(lh);
// WRF:7142
} else if((icdx==2)) {
// WRF:7145
cd = lake_max(0.45f,lake_min(1.0f,(0.45f+lake_div((0.35f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-500.f))))));
// WRF:7147
} else if((icdx==3)) {
// WRF:7149
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-170.0f))))));
// WRF:7150
} else if((icdx==4)) {
// WRF:7151
cd = lake_max(cdhmin,lake_min(cdhmax,(cdhmin+lake_div((((cdhmax-cdhmin))*((cdhdnmax-lake_max(cdhdnmin,lake_min(cdhdnmax,xdn(mgs,lh)))))),((cdhdnmax-cdhdnmin))))));
// WRF:7153
} else if((icdx==5)) {
// WRF:7154
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(mgs,lh),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7155
} else if((icdx==6)) {
// WRF:7156
indxr = (((int)(lake_div(((xdn(mgs,lh)-50.f)),100.f)))+1);
// WRF:7157
indxr = lake_min(ngdnmm,lake_max(1,indxr));
// WRF:7160
delrho = lake_max(0.0f,(0.01f*((xdn(mgs,lh)-coeff_mmgraupvt{}(indxr,1)))));
// WRF:7161
if((indxr<ngdnmm)) {
// WRF:7163
axx(mgs,lh) = (coeff_mmgraupvt{}(indxr,2)+(delrho*((coeff_mmgraupvt{}((indxr+1),2)-coeff_mmgraupvt{}(indxr,2)))));
// WRF:7164
bxx(mgs,lh) = (coeff_mmgraupvt{}(indxr,3)+(delrho*((coeff_mmgraupvt{}((indxr+1),3)-coeff_mmgraupvt{}(indxr,3)))));
// WRF:7167
} else {
// WRF:7168
axx(mgs,lh) = coeff_mmgraupvt{}(indxr,2);
// WRF:7169
bxx(mgs,lh) = coeff_mmgraupvt{}(indxr,3);
// WRF:7170
}
// WRF:7172
aax = axx(mgs,lh);
// WRF:7173
bbx = bxx(mgs,lh);
// WRF:7175
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-170.0f))))));
// WRF:7177
} else if((icdx<=0)) {
// WRF:7178
aax = coeff_ax{}(lh);
// WRF:7179
bbx = coeff_bx{}(lh);
// WRF:7180
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-170.0f))))));
// WRF:7181
} else {
// WRF:7182
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-170.0f))))));
// WRF:7183
}
// WRF:7185
cdxgs(mgs,lh) = cd;
// WRF:7186
if((((alpha(mgs,lh)==0.0f)&&(icdx>0))&&(icdx!=6))) {
// WRF:7190
axx(mgs,lh) = sqrtf(lake_div(((4.0f*xdn(mgs,lh))*gr),(((3.0f*cd)*rho00))));
// WRF:7191
bxx(mgs,lh) = 0.5f;
// WRF:7192
vtxbar(mgs,lh,1) = ((((lake_div(gf4p5,6.0f))*rhovt(mgs))*axx(mgs,lh))*sqrtf(xdia(mgs,lh,1)));
// WRF:7196
} else {
// WRF:7197
if((icdx!=6)) {bbx = coeff_bx{}(lh);}
// WRF:7198
tmp = ((4.f+alpha(mgs,lh))+bbx);
// WRF:7199
i = ((int)((dgami*(tmp))));
// WRF:7200
del = (tmp-(dgam*i));
// WRF:7201
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7203
tmp = (4.f+alpha(mgs,lh));
// WRF:7204
i = ((int)((dgami*(tmp))));
// WRF:7205
del = (tmp-(dgam*i));
// WRF:7206
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7211
if(((icdx>0)&&(icdx!=6))) {
// WRF:7212
aax = sqrtf(lake_div(((4.0f*xdn(mgs,lh))*gr),(((3.0f*cd)*rho00))));
// WRF:7213
vtxbar(mgs,lh,1) = lake_div((((rhovt(mgs)*aax)*sqrtf(xdia(mgs,lh,1)))*x),y);
// WRF:7214
axx(mgs,lh) = aax;
// WRF:7215
bxx(mgs,lh) = bbx;
// WRF:7216
} else if((icdx==6)) {
// WRF:7217
vtxbar(mgs,lh,1) = lake_div((((rhovt(mgs)*aax)*lake_pow(xdia(mgs,lh,1),bbx))*x),y);
// WRF:7218
} else {
// WRF:7219
axx(mgs,lh) = coeff_ax{}(lh);
// WRF:7220
bxx(mgs,lh) = coeff_bx{}(lh);
// WRF:7221
vtxbar(mgs,lh,1) = lake_div(((rhovt(mgs)*coeff_ax{}(lh))*((lake_pow(xdia(mgs,lh,1),coeff_bx{}(lh))*x))),y);
// WRF:7222
}
// WRF:7225
}
// WRF:7227
if((lwsm6&&(ipconc==0))) {
// WRF:7229
vtxbar(mgs,lh,1) = (((lake_div((330.f*gf4br),6.0f))*(lake_pow(xdia(mgs,lh,1),br)))*rhovt(mgs));
// WRF:7230
}
// WRF:7232
}
// WRF:7233
}
// WRF:7234
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7236
}
// WRF:7243
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:7245
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7246
vtxbar(mgs,lhl,1) = 0.0f;
// WRF:7247
if((qx(mgs,lhl)>coeff_qxmin{}(lhl))) {
// WRF:7249
if((icdxhl==1)) {
// WRF:7250
cd = coeff_cdx{}(lhl);
// WRF:7251
} else if((icdxhl==3)) {
// WRF:7253
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(mgs,lhl)))))),((800.f-170.0f))))));
// WRF:7254
} else if((icdxhl==4)) {
// WRF:7255
cd = lake_max(cdhlmin,lake_min(cdhlmax,(cdhlmin+lake_div((((cdhlmax-cdhlmin))*((cdhldnmax-lake_max(cdhldnmin,lake_min(cdhldnmax,xdn(mgs,lhl)))))),((cdhldnmax-cdhldnmin))))));
// WRF:7257
} else if((icdxhl==5)) {
// WRF:7258
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(mgs,lhl),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7259
} else if((icdxhl==6)) {
// WRF:7260
indxr = (((int)(lake_div(((xdn(mgs,lhl)-50.f)),100.f)))+1);
// WRF:7261
indxr = lake_min(ngdnmm,lake_max(1,indxr));
// WRF:7264
delrho = lake_max(0.0f,(0.01f*((xdn(mgs,lhl)-coeff_mmgraupvt{}(indxr,1)))));
// WRF:7265
if((indxr<ngdnmm)) {
// WRF:7267
axx(mgs,lhl) = (coeff_mmgraupvt{}(indxr,2)+(delrho*((coeff_mmgraupvt{}((indxr+1),2)-coeff_mmgraupvt{}(indxr,2)))));
// WRF:7268
bxx(mgs,lhl) = (coeff_mmgraupvt{}(indxr,3)+(delrho*((coeff_mmgraupvt{}((indxr+1),3)-coeff_mmgraupvt{}(indxr,3)))));
// WRF:7271
} else {
// WRF:7272
axx(mgs,lhl) = coeff_mmgraupvt{}(indxr,2);
// WRF:7273
bxx(mgs,lhl) = coeff_mmgraupvt{}(indxr,3);
// WRF:7274
}
// WRF:7276
aax = axx(mgs,lhl);
// WRF:7277
bbx = bxx(mgs,lhl);
// WRF:7279
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(mgs,lhl)))))),((800.f-170.0f))))));
// WRF:7281
} else {
// WRF:7285
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(mgs,lhl)))))),((800.f-170.0f))))));
// WRF:7286
}
// WRF:7288
cdxgs(mgs,lhl) = cd;
// WRF:7290
if((((alpha(mgs,lhl)==0.0f)&&(icdxhl>0))&&(icdxhl!=6))) {
// WRF:7294
axx(mgs,lhl) = sqrtf(lake_div(((4.0f*xdn(mgs,lhl))*gr),(((3.0f*cd)*rho00))));
// WRF:7295
bxx(mgs,lhl) = 0.5f;
// WRF:7296
vtxbar(mgs,lhl,1) = ((((lake_div(gf4p5,6.0f))*rhovt(mgs))*axx(mgs,lhl))*sqrtf(xdia(mgs,lhl,1)));
// WRF:7297
} else {
// WRF:7298
if((icdxhl!=6)) {bbx = coeff_bx{}(lhl);}
// WRF:7299
tmp = ((4.f+alpha(mgs,lhl))+bbx);
// WRF:7300
i = ((int)((dgami*(tmp))));
// WRF:7301
del = (tmp-(dgam*i));
// WRF:7302
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7304
tmp = (4.f+alpha(mgs,lhl));
// WRF:7305
i = ((int)((dgami*(tmp))));
// WRF:7306
del = (tmp-(dgam*i));
// WRF:7307
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7309
if(((icdxhl>0)&&(icdxhl!=6))) {
// WRF:7310
aax = sqrtf(lake_div(((4.0f*xdn(mgs,lhl))*gr),(((3.0f*cd)*rho00))));
// WRF:7311
vtxbar(mgs,lhl,1) = lake_div((((rhovt(mgs)*aax)*sqrtf(xdia(mgs,lhl,1)))*x),y);
// WRF:7312
axx(mgs,lhl) = aax;
// WRF:7313
bxx(mgs,lhl) = bbx;
// WRF:7314
} else if((icdxhl==6)) {
// WRF:7315
vtxbar(mgs,lhl,1) = lake_div((((rhovt(mgs)*aax)*lake_pow((xdia(mgs,lhl,1)),bbx))*x),y);
// WRF:7316
} else {
// WRF:7317
axx(mgs,lhl) = coeff_ax{}(lhl);
// WRF:7318
bxx(mgs,lhl) = coeff_bx{}(lhl);
// WRF:7319
vtxbar(mgs,lhl,1) = lake_div((rhovt(mgs)*(((coeff_ax{}(lhl)*lake_pow(xdia(mgs,lhl,1),coeff_bx{}(lhl)))*x))),y);
// WRF:7320
}
// WRF:7323
}
// WRF:7326
}
// WRF:7327
}
// WRF:7328
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7330
}
// WRF:7333
if((infdo>=1)) {
// WRF:7337
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7338
if(((ildo==0)||(ildo==lc))) {
// WRF:7339
vtxbar(mgs,lc,2) = vtxbar(mgs,lc,1);
// WRF:7340
}
// WRF:7341
if((li>1)) {
// WRF:7352
}
// WRF:7354
}
// WRF:7356
if((lg>lr)) {
// WRF:7358
#pragma unroll
for(il=lg;il<=lhab;il+=1) {
// WRF:7359
if(((ildo==0)||(ildo==il))) {
// WRF:7361
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7362
if((qx(mgs,il)>coeff_qxmin{}(il))) {
// WRF:7363
if(((((il==lh)&&(hssflg==1)))||((((lhl>1)&&(il==lhl))&&(hlssflg==1))))) {
// WRF:7368
if((il==lh)) {
// WRF:7370
if((icdx==1)) {
// WRF:7371
cd = coeff_cdx{}(lh);
// WRF:7372
} else if((icdx==2)) {
// WRF:7375
cd = lake_max(0.45f,lake_min(1.0f,(0.45f+lake_div((0.35f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-500.f))))));
// WRF:7377
} else if((icdx==3)) {
// WRF:7379
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(mgs,lh)))))),((800.f-170.0f))))));
// WRF:7380
} else if((icdx==4)) {
// WRF:7381
cd = lake_max(cdhmin,lake_min(cdhmax,(cdhmin+lake_div((((cdhmax-cdhmin))*((cdhdnmax-lake_max(cdhdnmin,lake_min(cdhdnmax,xdn(mgs,lh)))))),((cdhdnmax-cdhdnmin))))));
// WRF:7383
} else if((icdx==5)) {
// WRF:7384
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(mgs,lh),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7385
} else if((icdx==6)) {
// WRF:7386
aax = axx(mgs,lh);
// WRF:7387
bbx = bxx(mgs,lh);
// WRF:7388
} else if((icdx<=0)) {
// WRF:7389
aax = coeff_ax{}(lh);
// WRF:7390
bbx = coeff_bx{}(lh);
// WRF:7391
}
// WRF:7393
} else if(((lhl>1)&&(il==lhl))) {
// WRF:7395
if((icdxhl==1)) {
// WRF:7396
cd = coeff_cdx{}(lhl);
// WRF:7397
} else if((icdxhl==3)) {
// WRF:7399
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(mgs,lhl)))))),((800.f-170.0f))))));
// WRF:7400
} else if((icdxhl==4)) {
// WRF:7401
cd = lake_max(cdhlmin,lake_min(cdhlmax,(cdhlmin+lake_div((((cdhlmax-cdhlmin))*((cdhldnmax-lake_max(cdhldnmin,lake_min(cdhldnmax,xdn(mgs,lhl)))))),((cdhldnmax-cdhldnmin))))));
// WRF:7403
} else if((icdxhl==5)) {
// WRF:7406
cd = lake_max(0.45f,lake_min(0.6f,(0.45f+lake_div((0.15f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(mgs,lhl)))))),((800.f-500.f))))));
// WRF:7407
} else if((icdxhl==6)) {
// WRF:7408
aax = axx(mgs,lhl);
// WRF:7409
bbx = bxx(mgs,lhl);
// WRF:7410
} else if((icdxhl<=0)) {
// WRF:7411
aax = coeff_ax{}(lhl);
// WRF:7412
bbx = coeff_bx{}(lhl);
// WRF:7413
}
// WRF:7415
}
// WRF:7417
if((((alpha(mgs,il)==0.f)&&(infdo<2))&&((((((il==lh)&&(icdx>0))&&(icdx!=6)))||((((il==lhl)&&(icdxhl>0))&&(icdxhl!=6))))))) {
// WRF:7419
vtxbar(mgs,il,2) = sqrtf(lake_div(((((xdn(mgs,il)*xdia(mgs,il,1))*pi)*gr)),(((3.0f*cd)*lake_max(0.05f,rho0(mgs))))));
// WRF:7423
} else {
// WRF:7424
if(((il==lh)&&(icdx!=6))) {bbx = coeff_bx{}(il);}
// WRF:7425
if(((il==lhl)&&(icdxhl!=6))) {bbx = coeff_bx{}(il);}
// WRF:7426
tmp = ((1.f+alpha(mgs,il))+bbx);
// WRF:7427
i = ((int)((dgami*(tmp))));
// WRF:7428
del = (tmp-(dgam*i));
// WRF:7429
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7431
tmp = (1.f+alpha(mgs,il));
// WRF:7432
i = ((int)((dgami*(tmp))));
// WRF:7433
del = (tmp-(dgam*i));
// WRF:7434
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7436
if(((il==lh)||(il==lhl))) {
// WRF:7437
if((((il==lh)&&(icdx>0)))) {
// WRF:7438
if((icdx!=6)) {
// WRF:7439
aax = sqrtf(lake_div(((4.0f*xdn(mgs,il))*gr),(((3.0f*cd)*rho00))));
// WRF:7440
vtxbar(mgs,il,2) = lake_div((((rhovt(mgs)*aax)*lake_pow(xdia(mgs,il,1),coeff_bx{}(il)))*x),y);
// WRF:7441
} else {
// WRF:7442
vtxbar(mgs,il,2) = lake_div((((rhovt(mgs)*aax)*lake_pow(xdia(mgs,il,1),bbx))*x),y);
// WRF:7443
}
// WRF:7445
} else if((((il==lhl)&&(icdxhl>0)))) {
// WRF:7446
if((icdxhl!=6)) {
// WRF:7447
aax = sqrtf(lake_div(((4.0f*xdn(mgs,il))*gr),(((3.0f*cd)*rho00))));
// WRF:7448
vtxbar(mgs,il,2) = lake_div((((rhovt(mgs)*aax)*lake_pow(xdia(mgs,il,1),coeff_bx{}(il)))*x),y);
// WRF:7449
} else {
// WRF:7450
vtxbar(mgs,il,2) = lake_div((((rhovt(mgs)*aax)*lake_pow(xdia(mgs,il,1),bbx))*x),y);
// WRF:7451
}
// WRF:7452
} else {
// WRF:7453
aax = coeff_ax{}(il);
// WRF:7454
vtxbar(mgs,il,2) = lake_div(((rhovt(mgs)*coeff_ax{}(il))*((lake_pow(xdia(mgs,il,1),coeff_bx{}(il))*x))),y);
// WRF:7455
}
// WRF:7462
if((infdo>=2)) {
// WRF:7464
tmp = ((7.f+alpha(mgs,il))+bbx);
// WRF:7465
i = ((int)((dgami*(tmp))));
// WRF:7466
del = (tmp-(dgam*i));
// WRF:7467
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7469
tmp = (7.f+alpha(mgs,il));
// WRF:7470
i = ((int)((dgami*(tmp))));
// WRF:7471
del = (tmp-(dgam*i));
// WRF:7472
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7474
vtxbar(mgs,il,3) = lake_div((rhovt(mgs)*(((aax*lake_pow((xdia(mgs,il,1)),bbx))*x))),y);
// WRF:7478
if(((!(((vtxbar(mgs,il,1)>(-1.f))&&(vtxbar(mgs,il,1)<200.f))))||(!(((vtxbar(mgs,il,3)>(-1.f))&&(vtxbar(mgs,il,3)<200.f)))))) {
// WRF:7480
; /* native diagnostic only */
// WRF:7481
; /* native diagnostic only */
// WRF:7483
}
// WRF:7486
}
// WRF:7488
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7490
} else {
// WRF:7491
vtxbar(mgs,il,2) = lake_div((rhovt(mgs)*(((coeff_ax{}(il)*lake_pow(xdia(mgs,il,1),coeff_bx{}(il)))*x))),y);
// WRF:7495
if((infdo>=2)) {
// WRF:7496
vtxbar(mgs,il,3) = lake_div((rhovt(mgs)*(((aax*lake_pow((lake_div(1.0f,xdia(mgs,il,1))),((-bbx))))*gamma_sp(((7.0f+alpha(mgs,il))+bbx))))),gamma_sp((7.f+alpha(mgs,il))));
// WRF:7501
}
// WRF:7503
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7505
}
// WRF:7507
}
// WRF:7518
} else if(((((il==lh)&&(hssflg==0)))||((((lhl>1)&&(il==lhl))&&(hlssflg==0))))) {
// WRF:7519
vtxbar(mgs,il,2) = vtxbar(mgs,il,1);
// WRF:7520
vtxbar(mgs,il,3) = vtxbar(mgs,il,1);
// WRF:7521
} else {
// WRF:7522
vtxbar(mgs,il,2) = sqrtf(lake_div(((((xdn(mgs,il)*xdia(mgs,il,1))*pi)*gr)),(((3.0f*coeff_cdx{}(il))*lake_max(0.05f,rho0(mgs))))));
// WRF:7525
vtxbar(mgs,il,3) = vtxbar(mgs,il,1);
// WRF:7527
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7530
}
// WRF:7531
} else {
// WRF:7532
vtxbar(mgs,il,2) = 0.0f;
// WRF:7534
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7536
}
// WRF:7537
}
// WRF:7539
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7541
}
// WRF:7542
}
// WRF:7544
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7546
}
// WRF:7551
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7560
}
// WRF:7562
if(((lh>0)&&(graupelfallfac!=1.0f))) {
// WRF:7563
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7564
vtxbar(mgs,lh,1) = (graupelfallfac*vtxbar(mgs,lh,1));
// WRF:7565
vtxbar(mgs,lh,2) = (graupelfallfac*vtxbar(mgs,lh,2));
// WRF:7566
vtxbar(mgs,lh,3) = (graupelfallfac*vtxbar(mgs,lh,3));
// WRF:7567
axx(mgs,lh) = (graupelfallfac*axx(mgs,lh));
// WRF:7568
}
// WRF:7569
}
// WRF:7571
if(((lhl>0)&&(hailfallfac!=1.0f))) {
// WRF:7572
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7573
vtxbar(mgs,lhl,1) = (hailfallfac*vtxbar(mgs,lhl,1));
// WRF:7574
vtxbar(mgs,lhl,2) = (hailfallfac*vtxbar(mgs,lhl,2));
// WRF:7575
vtxbar(mgs,lhl,3) = (hailfallfac*vtxbar(mgs,lhl,3));
// WRF:7576
axx(mgs,lhl) = (hailfallfac*axx(mgs,lhl));
// WRF:7577
}
// WRF:7578
}
// WRF:7580
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7584
return;
}
__device__ __forceinline__ void setvtz_point(int ngscnt,Array<const float> qx,Array<const float> qxmin,Array<const float> qxw,Array<float> cx,Array<const float> rho0,Array<const float> rhovt,Array<float> xdia,Array<const float> cno,Array<const float> cnostmp,Array<float> xmas,Array<float> vtxbar,Array<float> xdn,Array<const float> xvmn0,Array<const float> xvmx0,Array<float> xv,Array<const float> cdx,Array<float> cdxgs,int ipconc1,int ndebug1,int ngs,int nz,Array<const int> kgs,Array<const float> fadvisc,float cwmasn,float cwmasx,float cwradn,Array<const float> cnina,float cimna,float cimxa,int itype1a,int itype2a,Array<const float> temcg,int infdo,Array<const float> alpha,int ildo,Array<float> axx,Array<float> bxx) {
int ngs0;
float cwc1;
float fsw;
float swmasmx;
float dtmp;
float cd;
float cwc0;
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwch{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwchl{};
float cwchtmp;
float cwchltmp;
float xnutmp;
float pii;
float cimasx;
float cimasn;
float cwrad;
float vr;
float rnux;
float alp;
float ccimx;

float arx;
float frx;
float vtrain;
float fw;
float fwlo;
float fwhi;
float rfwdiff;
float ar;
float br;
float cs;
float ds;
float gr;
float rwrad;
float rwdia;
float mwfac;
int il;
const float bta1=0.6f;
const float cnit=1.0e-02f;
float x;
float y;
float tmp;
float del;
float aax;
float bbx;
float delrho;
int indxr;
float mwt;
float nwt;
float zwt;
const float rho00=1.225f;
int i;
float xvbarmax;
int l1;
int l2;
// WRF:6312
fwlo = 0.2f;
// WRF:6313
fwhi = 0.4f;
// WRF:6314
rfwdiff = lake_div(1.f,((fwhi-fwlo)));
// WRF:6317
pii = piinv;
// WRF:6319
arx = 10.f;
// WRF:6320
frx = 516.575f;
// WRF:6322
ar = 841.99666f;
// WRF:6323
br = 0.8f;
// WRF:6324
gr = 9.8f;
// WRF:6326
cs = 12.42f;
// WRF:6327
ds = 0.42f;
// WRF:6329
if((ildo==0)) {
// WRF:6330
l1 = lc;
// WRF:6331
l2 = lhab;
// WRF:6332
} else {
// WRF:6333
l1 = ildo;
// WRF:6334
l2 = ildo;
// WRF:6335
}
// WRF:6347
if((lh>1)) {
// WRF:6348
if((dmuh==1.0f)) {
// WRF:6349
cwchtmp = lake_pow((((((3.f+coeff_dnu{}(lh)))*((2.f+coeff_dnu{}(lh))))*((1.0f+coeff_dnu{}(lh))))),(lake_div((-1.f),3.f)));
// WRF:6350
} else {
// WRF:6351
cwchtmp = lake_div(((6.0f*pii)*gamma_sp(lake_div(((coeff_xnu{}(lh)+1.f)),coeff_xmu{}(lh)))),gamma_sp(lake_div(((coeff_xnu{}(lh)+2.f)),coeff_xmu{}(lh))));
// WRF:6352
}
// WRF:6353
}
// WRF:6354
if((lhl>1)) {
// WRF:6355
if((dmuhl==1.0f)) {
// WRF:6356
cwchltmp = lake_pow((((((3.f+coeff_dnu{}(lhl)))*((2.f+coeff_dnu{}(lhl))))*((1.0f+coeff_dnu{}(lhl))))),(lake_div((-1.f),3.f)));
// WRF:6357
} else {
// WRF:6358
cwchltmp = lake_div(((6.0f*pii)*gamma_sp(lake_div(((coeff_xnu{}(lhl)+1)),coeff_xmu{}(lhl)))),gamma_sp(lake_div(((coeff_xnu{}(lhl)+2)),coeff_xmu{}(lhl))));
// WRF:6359
}
// WRF:6360
}
// WRF:6362
if((ipconc<=5)) {
// WRF:6363
if((lh>1)) {for(int _s0=1;_s0<=ngscnt;++_s0) {cwch(_s0) = cwchtmp;}}
// WRF:6364
if((lhl>1)) {for(int _s0=1;_s0<=ngscnt;++_s0) {cwchl(_s0) = cwchltmp;}}
// WRF:6365
} else {
// WRF:6366
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6368
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:6369
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:6370
if((dmuh==1.0f)) {
// WRF:6371
cwch(1) = lake_pow((((((3.f+alpha(1,lh)))*((2.f+alpha(1,lh))))*((1.0f+alpha(1,lh))))),(lake_div((-1.f),3.f)));
// WRF:6372
} else {
// WRF:6373
xnutmp = lake_div(((alpha(1,lh)-2.0f)),3.0f);
// WRF:6374
cwch(1) = lake_div(((6.0f*pii)*gamma_sp(lake_div(((xnutmp+1.f)),coeff_xmu{}(lh)))),gamma_sp(lake_div(((xnutmp+2.f)),coeff_xmu{}(lh))));
// WRF:6375
}
// WRF:6376
} else {
// WRF:6377
cwch(1) = cwchtmp;
// WRF:6378
}
// WRF:6379
}
// WRF:6380
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:6381
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:6382
if((dmuhl==1.0f)) {
// WRF:6383
cwchl(1) = lake_pow((((((3.f+alpha(1,lhl)))*((2.f+alpha(1,lhl))))*((1.0f+alpha(1,lhl))))),(lake_div((-1.f),3.f)));
// WRF:6384
} else {
// WRF:6385
xnutmp = lake_div(((alpha(1,lhl)-2.0f)),3.0f);
// WRF:6386
cwchl(1) = lake_div(((6.0f*pii)*gamma_sp(lake_div(((xnutmp+1)),coeff_xmu{}(lhl)))),gamma_sp(lake_div(((xnutmp+2)),coeff_xmu{}(lhl))));
// WRF:6387
}
// WRF:6388
} else {
// WRF:6389
cwchl(1) = cwchltmp;
// WRF:6390
}
// WRF:6391
}
// WRF:6393
}
// WRF:6395
}
// WRF:6398
cimasn = lake_min(cimas0,6.88e-13f);
// WRF:6399
cimasx = 1.0e-8f;
// WRF:6400
ccimx = 5000.0e3f;
// WRF:6402
cwc1 = lake_div(6.0f,((pi*1000.f)));
// WRF:6403
cwc0 = pii;
// WRF:6404
mwfac = lake_pow(6.0f,(lake_div(1.f,3.f)));
// WRF:6407
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6418
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6420
if(((ildo==0)||(ildo==lc))) {
// WRF:6422
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6423
xv(1,lc) = 0.0f;
// WRF:6425
if((qx(1,lc)>coeff_qxmin{}(lc))) {
// WRF:6427
if((ipconc>=2)) {
// WRF:6428
if((cx(1,lc)>cxmin)) {
// WRF:6429
xmas(1,lc) = lake_min(lake_max(lake_div((qx(1,lc)*rho0(1)),cx(1,lc)),cwmasn),cwmasx);
// WRF:6431
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6432
} else {
// WRF:6433
cx(1,lc) = lake_max(cxmin,lake_div((rho0(1)*qx(1,lc)),cwmasx));
// WRF:6434
xmas(1,lc) = lake_min(lake_max(lake_div((qx(1,lc)*rho0(1)),cx(1,lc)),cwmasn),cwmasx);
// WRF:6435
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6437
}
// WRF:6438
} else {
// WRF:6439
if((ipconc<2)) {
// WRF:6440
cx(1,lc) = lake_div((rho0(1)*ccn),rho00);
// WRF:6441
}
// WRF:6442
if(((qx(1,lc)>coeff_qxmin{}(lc))&&(cx(1,lc)>0.01f))) {
// WRF:6443
xmas(1,lc) = lake_min(lake_max(lake_div((qx(1,lc)*rho0(1)),cx(1,lc)),(xdn(1,lc)*coeff_xvmn{}(lc))),(xdn(1,lc)*coeff_xvmx{}(lc)));
// WRF:6447
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6448
cx(1,lc) = lake_div((qx(1,lc)*rho0(1)),xmas(1,lc));
// WRF:6450
} else if(((qx(1,lc)>coeff_qxmin{}(lc))&&(cx(1,lc)<=1.0e-9f))) {
// WRF:6451
cx(1,lc) = lake_max(cxmin,lake_div((rho0(1)*qx(1,lc)),cwmasx));
// WRF:6452
xmas(1,lc) = lake_min(lake_max(lake_div((qx(1,lc)*rho0(1)),cx(1,lc)),cwmasn),cwmasx);
// WRF:6454
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6456
} else if(((qx(1,lc)>coeff_qxmin{}(lc))&&(cx(1,lc)<=0.01f))) {
// WRF:6457
xmas(1,lc) = (lake_div(((xdn(1,lc)*4.f)*pi),3.f)*lake_pow((5.0e-6f),3));
// WRF:6458
cx(1,lc) = lake_div((rho0(1)*qx(1,lc)),xmas(1,lc));
// WRF:6459
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6461
} else {
// WRF:6462
xmas(1,lc) = cwmasn;
// WRF:6463
xv(1,lc) = lake_div(xmas(1,lc),1000.f);
// WRF:6465
}
// WRF:6466
}
// WRF:6475
xdia(1,lc,1) = lake_pow(((xmas(1,lc)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6476
xdia(1,lc,2) = lake_pow(xdia(1,lc,1),2);
// WRF:6477
xdia(1,lc,3) = xdia(1,lc,1);
// WRF:6478
cwrad = (0.5f*xdia(1,lc,1));
// WRF:6479
if((fadvisc(1)>0.0f)) {
// WRF:6480
vtxbar(1,lc,1) = lake_div(((((2.0f*gr)*xdn(1,lc))*(lake_pow(cwrad,2)))),((9.0f*fadvisc(1))));
// WRF:6483
} else {
// WRF:6484
vtxbar(1,lc,1) = 0.0f;
// WRF:6485
}
// WRF:6488
} else {
// WRF:6489
xmas(1,lc) = cwmasn;
// WRF:6490
xv(1,lc) = lake_div(xmas(1,lc),xdn(1,lc));
// WRF:6491
if((qx(1,lc)<=0.0f)) {cx(1,lc) = 0.0f;}
// WRF:6492
if((ipconc<=1)) {cx(1,lc) = 0.01f;}
// WRF:6493
xdia(1,lc,1) = (2.f*cwradn);
// WRF:6494
xdia(1,lc,2) = (4.f*lake_pow(cwradn,2));
// WRF:6495
xdia(1,lc,3) = xdia(1,lc,1);
// WRF:6496
vtxbar(1,lc,1) = 0.0f;
// WRF:6498
}
// WRF:6500
}
// WRF:6502
}
// WRF:6514
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6516
if(((li>1)&&(((ildo==0)||(ildo==li))))) {
// WRF:6517
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6518
xdn(1,li) = 900.0f;
// WRF:6519
if((ipconc==0)) {
// WRF:6521
cx(1,li) = cnina(1);
// WRF:6522
if((cimna>1.0f)) {
// WRF:6523
cx(1,li) = lake_max(cimna,cx(1,li));
// WRF:6524
}
// WRF:6525
if((cimxa>1.0f)) {
// WRF:6526
cx(1,li) = lake_min(cimxa,cx(1,li));
// WRF:6527
}
// WRF:6529
if(((itype1a>=1)||(itype2a>=1))) {
// WRF:6530
cx(1,li) = lake_max(cx(1,li),lake_div((qx(1,li)*rho0(1)),cimasx));
// WRF:6531
cx(1,li) = lake_min(cx(1,li),lake_div((qx(1,li)*rho0(1)),cimasn));
// WRF:6532
}
// WRF:6534
cx(1,li) = lake_max(1.0e-20f,cx(1,li));
// WRF:6538
} else if((ipconc>=1)) {
// WRF:6539
if((qx(1,li)>coeff_qxmin{}(li))) {
// WRF:6540
cx(1,li) = lake_max(cx(1,li),lake_div((qx(1,li)*rho0(1)),cimasx));
// WRF:6541
cx(1,li) = lake_min(cx(1,li),lake_div((qx(1,li)*rho0(1)),cimasn));
// WRF:6543
}
// WRF:6544
}
// WRF:6546
if((qx(1,li)>coeff_qxmin{}(li))) {
// WRF:6547
xmas(1,li) = lake_max(lake_div((qx(1,li)*rho0(1)),cx(1,li)),cimasn);
// WRF:6554
if((xmas(1,li)>0.0f)) {
// WRF:6559
if((ixtaltype==1)) {
// WRF:6560
xdia(1,li,1) = (0.1871f*(lake_pow(xmas(1,li),(0.3429f))));
// WRF:6561
xdia(1,li,3) = (0.1871f*(lake_pow(xmas(1,li),(0.3429f))));
// WRF:6562
} else if((ixtaltype==2)) {
// WRF:6563
xdia(1,li,1) = (0.277823f*lake_pow(xmas(1,li),0.359971f));
// WRF:6564
xdia(1,li,3) = (0.277823f*lake_pow(xmas(1,li),0.359971f));
// WRF:6565
}
// WRF:6566
}
// WRF:6571
if((ipconc>=0)) {
// WRF:6574
xv(1,li) = lake_div(xmas(1,li),xdn(1,li));
// WRF:6575
if((icefallopt==1)) {
// WRF:6576
if((ixtaltype==1)) {
// WRF:6577
tmp = lake_div(((67056.6300748612f*rhovt(1))),((lake_pow((lake_div(((1.0f+cinu)),xv(1,li))),0.4716666666666667f)*gfcinu1)));
// WRF:6579
vtxbar(1,li,2) = (tmp*gfcinu1p47);
// WRF:6580
vtxbar(1,li,1) = lake_div((tmp*gfcinu2p47),((1.f+cinu)));
// WRF:6581
vtxbar(1,li,3) = vtxbar(1,li,1);
// WRF:6582
} else if((ixtaltype==2)) {
// WRF:6583
vtxbar(1,li,1) = ((11.9495f*rhovt(1))*lake_pow((xv(1,li)),(0.14f)));
// WRF:6584
vtxbar(1,li,2) = ((7.02909f*rhovt(1))*lake_pow((xv(1,li)),(0.14f)));
// WRF:6585
vtxbar(1,li,3) = vtxbar(1,li,1);
// WRF:6587
}
// WRF:6589
} else if((icefallopt==2)) {
// WRF:6590
tmp = lake_div(((82.3166f*rhovt(1))),((lake_pow((lake_div(((1.0f+cinu)),xv(1,li))),0.22117f)*gfcinu1)));
// WRF:6592
vtxbar(1,li,2) = (tmp*gfcinu1p22);
// WRF:6593
vtxbar(1,li,1) = lake_div((tmp*gfcinu2p22),((1.f+cinu)));
// WRF:6594
vtxbar(1,li,3) = vtxbar(1,li,1);
// WRF:6596
} else if((icefallopt==3)) {
// WRF:6598
tmp = lake_div(((47.6273f*rhovt(1))),((lake_pow((lake_div(((1.0f+cinu)),xv(1,li))),0.18333f)*gfcinu1)));
// WRF:6600
vtxbar(1,li,2) = (tmp*gfcinu1p18);
// WRF:6601
vtxbar(1,li,1) = lake_div((tmp*gfcinu2p18),((1.f+cinu)));
// WRF:6602
vtxbar(1,li,3) = vtxbar(1,li,1);
// WRF:6604
}
// WRF:6608
xdia(1,li,2) = lake_pow(xdia(1,li,1),2);
// WRF:6610
} else {
// WRF:6611
xdia(1,li,1) = lake_max(xdia(1,li,1),10.e-6f);
// WRF:6612
xdia(1,li,1) = lake_min(xdia(1,li,1),1000.e-6f);
// WRF:6613
vtxbar(1,li,1) = ((4.942e4f)*(lake_pow(xdia(1,li,1),(1.4150f))));
// WRF:6615
xdn(1,li) = 900.0f;
// WRF:6616
xdia(1,li,2) = lake_pow(xdia(1,li,1),2);
// WRF:6617
vtxbar(1,li,1) = (vtxbar(1,li,1)*rhovt(1));
// WRF:6618
xv(1,li) = lake_div(xmas(1,li),xdn(1,li));
// WRF:6619
}
// WRF:6620
} else {
// WRF:6621
xmas(1,li) = 1.e-13f;
// WRF:6622
if((qx(1,li)<=0.0f)) {cx(1,li) = 0.0f;}
// WRF:6623
xdn(1,li) = 900.0f;
// WRF:6624
xdia(1,li,1) = 1.e-7f;
// WRF:6625
xdia(1,li,2) = (1.e-14f);
// WRF:6626
xdia(1,li,3) = 1.e-7f;
// WRF:6627
vtxbar(1,li,1) = 0.0f;
// WRF:6630
}
// WRF:6632
if((icefallfac!=1.0f)) {
// WRF:6633
vtxbar(1,li,1) = (icefallfac*vtxbar(1,li,1));
// WRF:6634
vtxbar(1,li,2) = (icefallfac*vtxbar(1,li,2));
// WRF:6635
vtxbar(1,li,3) = (icefallfac*vtxbar(1,li,3));
// WRF:6636
}
// WRF:6640
}
// WRF:6642
}
// WRF:6651
if(((ildo==0)||(ildo==lr))) {
// WRF:6652
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6653
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:6658
if((ipconc>=3)) {
// WRF:6659
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*lake_max(1.0e-11f,cx(1,lr)))));
// WRF:6660
xvbarmax = coeff_xvmx{}(lr);
// WRF:6661
if((imaxdiaopt==1)) {
// WRF:6662
xvbarmax = coeff_xvmx{}(lr);
// WRF:6663
} else if((imaxdiaopt==2)) {
// WRF:6664
if((imurain==1)) {
// WRF:6665
xvbarmax = lake_div(coeff_xvmx{}(lr),(lake_div(lake_pow(((3.f+alpha(1,lr))),3),(((((3.f+alpha(1,lr)))*((2.f+alpha(1,lr))))*((1.f+alpha(1,lr))))))));
// WRF:6666
} else if((imurain==3)) {
// WRF:6668
}
// WRF:6669
} else if((imaxdiaopt==3)) {
// WRF:6670
if((imurain==1)) {
// WRF:6671
xvbarmax = lake_div(coeff_xvmx{}(lr),(lake_div(lake_pow(((4.f+alpha(1,lr))),3),(((((3.f+alpha(1,lr)))*((2.f+alpha(1,lr))))*((1.f+alpha(1,lr))))))));
// WRF:6672
} else if((imurain==3)) {
// WRF:6674
}
// WRF:6675
}
// WRF:6677
if((xv(1,lr)>xvbarmax)) {
// WRF:6678
xv(1,lr) = xvbarmax;
// WRF:6679
cx(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xvbarmax*xdn(1,lr))));
// WRF:6680
} else if((xv(1,lr)<coeff_xvmn{}(lr))) {
// WRF:6681
xv(1,lr) = coeff_xvmn{}(lr);
// WRF:6682
cx(1,lr) = lake_div((rho0(1)*qx(1,lr)),((coeff_xvmn{}(lr)*xdn(1,lr))));
// WRF:6683
}
// WRF:6686
xmas(1,lr) = (xv(1,lr)*xdn(1,lr));
// WRF:6687
xdia(1,lr,3) = lake_pow(((xmas(1,lr)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6688
if((imurain==3)) {
// WRF:6690
xdia(1,lr,1) = xdia(1,lr,3);
// WRF:6691
} else {
// WRF:6692
xdia(1,lr,1) = lake_pow((lake_div(((6.f*pii)*xv(1,lr)),(((((alpha(1,lr)+3.f))*((alpha(1,lr)+2.f)))*((alpha(1,lr)+1.f)))))),(lake_div(1.f,3.f)));
// WRF:6693
}
// WRF:6700
} else {
// WRF:6701
xdia(1,lr,1) = lake_pow((lake_div((qx(1,lr)*rho0(1)),(((pi*xdn(1,lr))*coeff_cno{}(lr))))),(0.25f));
// WRF:6703
xmas(1,lr) = ((xdn(1,lr)*(lake_div(pi,6.f)))*lake_pow(xdia(1,lr,1),3));
// WRF:6704
xdia(1,lr,3) = lake_pow(((xmas(1,lr)*cwc1)),(lake_div(1.f,3.f)));
// WRF:6705
cx(1,lr) = (coeff_cno{}(lr)*xdia(1,lr,1));
// WRF:6706
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*cx(1,lr))));
// WRF:6707
}
// WRF:6708
} else {
// WRF:6709
xdia(1,lr,1) = 1.e-9f;
// WRF:6710
xdia(1,lr,3) = 1.e-9f;
// WRF:6711
xmas(1,lr) = ((xdn(1,lr)*(lake_div(pi,6.f)))*lake_pow(xdia(1,lr,1),3));
// WRF:6713
}
// WRF:6714
xdia(1,lr,2) = lake_pow(xdia(1,lr,1),2);
// WRF:6716
}
// WRF:6718
}
// WRF:6724
if(((ls>1)&&(((ildo==0)||(ildo==ls))))) {
// WRF:6726
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6727
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:6728
if((ipconc>=4)) {
// WRF:6730
xmas(1,ls) = lake_div((rho0(1)*qx(1,ls)),(lake_max(1.0e-9f,cx(1,ls))));
// WRF:6731
swmasmx = 13.7e-6f;
// WRF:6737
if((isnowdens==2)) {
// WRF:6739
xdn(1,ls) = (0.0346159f*sqrtf(lake_div(cx(1,ls),((qx(1,ls)*rho0(1))))));
// WRF:6740
xdn(1,ls) = lake_max(100.0f,xdn(1,ls));
// WRF:6742
if((xdn(1,ls)<=900.f)) {
// WRF:6743
dtmp = sqrtf(lake_div(xmas(1,ls),0.069f));
// WRF:6744
xv(1,ls) = (28.8887f*lake_pow(xmas(1,ls),(lake_div(3.f,2.f))));
// WRF:6745
} else {
// WRF:6746
xdn(1,ls) = 900.f;
// WRF:6747
xv(1,ls) = lake_div((rho0(1)*qx(1,ls)),((xdn(1,ls)*lake_max(1.0e-9f,cx(1,ls)))));
// WRF:6748
dtmp = lake_pow((((xv(1,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6749
}
// WRF:6751
} else {
// WRF:6752
xv(1,ls) = lake_div((rho0(1)*qx(1,ls)),((xdn(1,ls)*lake_max(1.0e-9f,cx(1,ls)))));
// WRF:6753
dtmp = lake_pow((((xv(1,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6754
}
// WRF:6756
xdia(1,ls,1) = dtmp;
// WRF:6758
if(((xv(1,ls)<coeff_xvmn{}(ls))&&(isnowdens==1))) {
// WRF:6759
xv(1,ls) = lake_max(coeff_xvmn{}(ls),xv(1,ls));
// WRF:6760
xmas(1,ls) = (xv(1,ls)*xdn(1,ls));
// WRF:6761
cx(1,ls) = lake_div((rho0(1)*qx(1,ls)),(xmas(1,ls)));
// WRF:6762
xdia(1,ls,1) = lake_pow((((xv(1,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6763
}
// WRF:6765
if((xv(1,ls)>(coeff_xvmx{}(ls)*lake_max(1.f,lake_div(100.f,lake_min(100.f,xdn(1,ls))))))) {
// WRF:6766
xv(1,ls) = lake_min(coeff_xvmx{}(ls),lake_max(coeff_xvmn{}(ls),xv(1,ls)));
// WRF:6767
xmas(1,ls) = (0.106214f*lake_pow(xv(1,ls),(lake_div(2.f,3.f))));
// WRF:6768
cx(1,ls) = lake_div((rho0(1)*qx(1,ls)),(xmas(1,ls)));
// WRF:6769
xdn(1,ls) = (0.0346159f*sqrtf(lake_div(cx(1,ls),((qx(1,ls)*rho0(1))))));
// WRF:6770
xdia(1,ls,1) = sqrtf(lake_div(xmas(1,ls),0.069f));
// WRF:6771
}
// WRF:6773
xdia(1,ls,3) = xdia(1,ls,1);
// WRF:6775
} else {
// WRF:6776
xdia(1,ls,1) = lake_pow((lake_div((qx(1,ls)*rho0(1)),(((pi*xdn(1,ls))*cnostmp(1))))),(0.25f));
// WRF:6778
cx(1,ls) = (cnostmp(1)*xdia(1,ls,1));
// WRF:6779
xv(1,ls) = lake_div((rho0(1)*qx(1,ls)),((xdn(1,ls)*cx(1,ls))));
// WRF:6780
xdia(1,ls,3) = lake_pow((((xv(1,ls)*cwc0)*6.0f)),(lake_div(1.f,3.f)));
// WRF:6781
}
// WRF:6782
} else {
// WRF:6783
xdia(1,ls,1) = 1.e-9f;
// WRF:6784
xdia(1,ls,3) = 1.e-9f;
// WRF:6785
cx(1,ls) = 0.0f;
// WRF:6787
if((isnowdens==2)) {
// WRF:6788
xdn(1,ls) = 90.f;
// WRF:6789
}
// WRF:6791
}
// WRF:6792
xdia(1,ls,2) = lake_pow(xdia(1,ls,1),2);
// WRF:6795
}
// WRF:6797
}
// WRF:6805
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:6807
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6808
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:6809
if((ipconc>=5)) {
// WRF:6811
xv(1,lh) = lake_div((rho0(1)*qx(1,lh)),((xdn(1,lh)*lake_max(1.0e-9f,cx(1,lh)))));
// WRF:6812
xmas(1,lh) = (xv(1,lh)*xdn(1,lh));
// WRF:6814
if(((xv(1,lh)<coeff_xvmn{}(lh))||(xv(1,lh)>coeff_xvmx{}(lh)))) {
// WRF:6815
xv(1,lh) = lake_min(coeff_xvmx{}(lh),lake_max(coeff_xvmn{}(lh),xv(1,lh)));
// WRF:6816
xmas(1,lh) = (xv(1,lh)*xdn(1,lh));
// WRF:6817
cx(1,lh) = lake_div((rho0(1)*qx(1,lh)),(xmas(1,lh)));
// WRF:6818
}
// WRF:6820
xdia(1,lh,3) = lake_pow((((xv(1,lh)*6.f)*pii)),(lake_div(1.f,3.f)));
// WRF:6821
if((dmuh==1.0f)) {
// WRF:6822
xdia(1,lh,1) = (cwch(1)*xdia(1,lh,3));
// WRF:6823
} else {
// WRF:6824
xdia(1,lh,1) = lake_pow(((xv(1,lh)*cwch(1))),(lake_div(1.f,3.f)));
// WRF:6825
}
// WRF:6827
} else {
// WRF:6828
xdia(1,lh,1) = lake_pow((lake_div((qx(1,lh)*rho0(1)),(((pi*xdn(1,lh))*coeff_cno{}(lh))))),(0.25f));
// WRF:6830
cx(1,lh) = (coeff_cno{}(lh)*xdia(1,lh,1));
// WRF:6831
xv(1,lh) = lake_max(coeff_xvmn{}(lh),lake_div((rho0(1)*qx(1,lh)),((xdn(1,lh)*cx(1,lh)))));
// WRF:6832
xdia(1,lh,3) = lake_pow((lake_div((xv(1,lh)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6833
}
// WRF:6834
} else {
// WRF:6835
xdia(1,lh,1) = 1.e-9f;
// WRF:6836
xdia(1,lh,3) = 1.e-9f;
// WRF:6837
}
// WRF:6838
xdia(1,lh,2) = lake_pow(xdia(1,lh,1),2);
// WRF:6841
}
// WRF:6843
}
// WRF:6851
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:6853
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6854
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:6855
if((ipconc>=5)) {
// WRF:6857
xv(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),((xdn(1,lhl)*lake_max(1.0e-9f,cx(1,lhl)))));
// WRF:6858
xmas(1,lhl) = (xv(1,lhl)*xdn(1,lhl));
// WRF:6861
if(((xv(1,lhl)<coeff_xvmn{}(lhl))||(xv(1,lhl)>coeff_xvmx{}(lhl)))) {
// WRF:6862
xv(1,lhl) = lake_min(coeff_xvmx{}(lhl),lake_max(coeff_xvmn{}(lhl),xv(1,lhl)));
// WRF:6863
xmas(1,lhl) = (xv(1,lhl)*xdn(1,lhl));
// WRF:6864
cx(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),(xmas(1,lhl)));
// WRF:6865
}
// WRF:6867
xdia(1,lhl,3) = lake_pow((lake_div((xv(1,lhl)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6868
if((dmuhl==1.0f)) {
// WRF:6869
xdia(1,lhl,1) = (cwchl(1)*xdia(1,lhl,3));
// WRF:6870
} else {
// WRF:6871
xdia(1,lhl,1) = lake_pow(((xv(1,lhl)*cwchl(1))),(lake_div(1.f,3.f)));
// WRF:6872
}
// WRF:6875
} else {
// WRF:6876
xdia(1,lhl,1) = lake_pow((lake_div((qx(1,lhl)*rho0(1)),(((pi*xdn(1,lhl))*coeff_cno{}(lhl))))),(0.25f));
// WRF:6878
cx(1,lhl) = (coeff_cno{}(lhl)*xdia(1,lhl,1));
// WRF:6879
xv(1,lhl) = lake_max(coeff_xvmn{}(lhl),lake_div((rho0(1)*qx(1,lhl)),((xdn(1,lhl)*cx(1,lhl)))));
// WRF:6880
xdia(1,lhl,3) = lake_pow((lake_div((xv(1,lhl)*6.f),pi)),(lake_div(1.f,3.f)));
// WRF:6881
}
// WRF:6882
} else {
// WRF:6883
xdia(1,lhl,1) = 1.e-9f;
// WRF:6884
xdia(1,lhl,3) = 1.e-9f;
// WRF:6885
}
// WRF:6886
xdia(1,lhl,2) = lake_pow(xdia(1,lhl,1),2);
// WRF:6889
}
// WRF:6891
}
// WRF:6907
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:6914
if(((ildo==0)||(ildo==lr))) {
// WRF:6915
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:6916
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:6917
if((ipconc<3)) {
// WRF:6918
vtxbar(1,lr,1) = (((rainfallfac*(lake_div((ar*gf4br),6.0f)))*(lake_pow(xdia(1,lr,1),br)))*rhovt(1));
// WRF:6920
} else {
// WRF:6922
if((imurain==1)) {
// WRF:6929
alp = alpha(1,lr);
// WRF:6931
vtxbar(1,lr,1) = ((rhovt(1)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(1,lr,1)))),(((-alp)-4.0f))))));
// WRF:6933
if(((infdo>=1)&&(rssflg==1))) {
// WRF:6934
vtxbar(1,lr,2) = ((rhovt(1)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(1,lr,1)))),(((-alp)-1.0f))))));
// WRF:6935
} else {
// WRF:6936
vtxbar(1,lr,2) = vtxbar(1,lr,1);
// WRF:6937
}
// WRF:6939
if(((infdo>=2)&&(rssflg==1))) {
// WRF:6940
vtxbar(1,lr,3) = ((rhovt(1)*arx)*((1.0f-lake_pow(((1.0f+(frx*xdia(1,lr,1)))),(((-alp)-7.0f))))));
// WRF:6941
} else {
// WRF:6942
vtxbar(1,lr,3) = vtxbar(1,lr,1);
// WRF:6943
}
// WRF:6947
} else if((imurain==3)) {
// WRF:6949
if((lzr<1)) {
// WRF:6950
rwdia = lake_min(xdia(1,lr,1),8.0e-3f);
// WRF:6952
vtxbar(1,lr,1) = (((rhovt(1)*6.0f)*pii)*(((((0.04771f+(3788.0f*rwdia))-(1.105e6f*lake_pow(rwdia,2)))+(1.412e8f*lake_pow(rwdia,3)))-(6.527e9f*lake_pow(rwdia,4)))));
// WRF:6955
if((infdo>=1)) {
// WRF:6956
if((rssflg>=1)) {
// WRF:6957
vtxbar(1,lr,2) = ((((((0.09112f+(2714.0f*rwdia))-(4.872e5f*lake_pow(rwdia,2)))+(4.495e7f*lake_pow(rwdia,3)))-(1.626e9f*lake_pow(rwdia,4))))*rhovt(1));
// WRF:6959
} else {
// WRF:6960
vtxbar(1,lr,2) = vtxbar(1,lr,1);
// WRF:6961
}
// WRF:6962
}
// WRF:6964
if((infdo>=2)) {
// WRF:6965
vtxbar(1,lr,3) = (rhovt(1)*(((((0.0911229f+(9246.494f*(rwdia)))-(3.2839926e6f*(lake_pow(rwdia,2))))+(4.944093e8f*(lake_pow(rwdia,3))))-(2.631718e10f*(lake_pow(rwdia,4))))));
// WRF:6971
}
// WRF:6973
} else {
// WRF:6975
vr = xv(1,lr);
// WRF:6976
rnux = alpha(1,lr);
// WRF:6978
if(((infdo>=1)&&(rssflg==1))) {
// WRF:6979
vtxbar(1,lr,2) = lake_div((rhovt(1)*((lake_pow((lake_div(((1.f+rnux)),vr)),((-1.333333f)))*(((((((0.0911229f*lake_pow((lake_div(((1.f+rnux)),vr)),1.333333f))*gamma_sp((1.f+rnux)))+lake_div((((5430.3131f*((1.f+rnux)))*gamma_sp((lake_div(4.f,3.f)+rnux)))),vr))-((1.0732802e6f*lake_pow((lake_div(((1.f+rnux)),vr)),0.6666667f))*gamma_sp((1.666667f+rnux))))+((8.584110982429507e7f*lake_pow((lake_div(((1.f+rnux)),vr)),(lake_div(1.f,3.f))))*gamma_sp((2.f+rnux))))-(2.3303765697228556e9f*gamma_sp((lake_div(7.f,3.f)+rnux)))))))),gamma_sp((1.f+rnux)));
// WRF:6989
}
// WRF:6992
vtxbar(1,lr,1) = lake_div((rhovt(1)*(((((((0.0911229f*lake_pow(((1+rnux)),1.3333333333333333f))*gamma_sp((2.f+rnux)))+(((5430.313059683277f*((1+rnux)))*lake_pow(vr,0.3333333333333333f))*gamma_sp((2.333333333333333f+rnux))))-(((1.0732802065650471e6f*lake_pow(((1+rnux)),0.6666666666666666f))*lake_pow(vr,0.6666666666666666f))*gamma_sp((2.6666666666666667f+rnux))))+(((8.584110982429507e7f*lake_pow(((1+rnux)),0.3333333333333333f))*vr)*gamma_sp((3+rnux))))-((2.3303765697228556e9f*lake_pow(vr,1.3333333333333333f))*gamma_sp((3.333333333333333f+rnux)))))),((lake_pow(((1+rnux)),2.333333333333333f)*gamma_sp((1+rnux)))));
// WRF:7003
if(((infdo>=1)&&(rssflg==0))) {
// WRF:7004
vtxbar(1,lr,2) = vtxbar(1,lr,1);
// WRF:7005
}
// WRF:7007
if(((infdo>=2)&&(rssflg==1))) {
// WRF:7008
vtxbar(1,lr,3) = lake_div((rhovt(1)*((((1.f+rnux))*(((((((0.0911229f*lake_pow(((1+rnux)),1.3333333333333333f))*gamma_sp((3.f+rnux)))+(((5430.313059683277f*((1+rnux)))*lake_pow(vr,0.3333333333333333f))*gamma_sp((3.3333333333333335f+rnux))))-(((1.0732802065650471e6f*lake_pow(((1+rnux)),0.6666666666666666f))*lake_pow(vr,0.6666666666666666f))*gamma_sp((3.6666666666666665f+rnux))))+(((8.5841109824295e7f*lake_pow(((1+rnux)),0.3333333333333333f))*vr)*gamma_sp((4.f+rnux))))-((2.3303765697228556e9f*lake_pow(vr,1.3333333333333333f))*gamma_sp((4.333333333333333f+rnux)))))))),(((lake_pow(((1+rnux)),3.3333333333333335f)*((2+rnux)))*gamma_sp((1+rnux)))));
// WRF:7022
} else if((infdo>=2)) {
// WRF:7023
vtxbar(1,lr,3) = vtxbar(1,lr,1);
// WRF:7024
}
// WRF:7027
}
// WRF:7028
}
// WRF:7040
}
// WRF:7041
} else {
// WRF:7042
vtxbar(1,lr,1) = 0.0f;
// WRF:7043
vtxbar(1,lr,2) = 0.0f;
// WRF:7044
}
// WRF:7045
}
// WRF:7046
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7048
}
// WRF:7054
if(((ls>1)&&(((ildo==0)||(ildo==ls))))) {
// WRF:7055
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7056
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:7057
if((ipconc>=4)) {
// WRF:7058
if((mixedphase&&qsvtmod)) {
// WRF:7059
} else {
// WRF:7060
if((isnowfall==1)) {
// WRF:7062
vtxbar(1,ls,1) = ((5.72462f*rhovt(1))*lake_pow((xv(1,ls)),(lake_div(1.f,12.f))));
// WRF:7063
} else if((isnowfall==2)) {
// WRF:7065
if((isnowdens==1)) {
// WRF:7066
vtxbar(1,ls,1) = ((11.9495f*rhovt(1))*lake_pow((xv(1,ls)),(0.14f)));
// WRF:7067
} else {
// WRF:7068
vtxbar(1,ls,1) = ((11.9495f*rhovt(1))*lake_pow((lake_div((xv(1,ls)*xdn(1,ls)),100.f)),(0.14f)));
// WRF:7069
}
// WRF:7070
} else if((isnowfall==3)) {
// WRF:7072
vtxbar(1,ls,1) = ((50.092f*rhovt(1))*lake_pow((xmas(1,ls)),(0.2635f)));
// WRF:7073
}
// WRF:7075
if((lake_abs(sssflg)>=1)) {
// WRF:7076
if((isnowfall==1)) {
// WRF:7077
vtxbar(1,ls,2) = ((4.04091f*rhovt(1))*lake_pow((xv(1,ls)),(lake_div(1.f,12.f))));
// WRF:7078
} else if((isnowfall==2)) {
// WRF:7080
if((isnowdens==1)) {
// WRF:7081
vtxbar(1,ls,2) = ((7.02909f*rhovt(1))*lake_pow((xv(1,ls)),(0.14f)));
// WRF:7082
} else {
// WRF:7083
vtxbar(1,ls,2) = ((7.02909f*rhovt(1))*lake_pow((lake_div((xv(1,ls)*xdn(1,ls)),100.f)),(0.14f)));
// WRF:7084
}
// WRF:7085
} else if((isnowfall==3)) {
// WRF:7087
vtxbar(1,ls,2) = ((21.6147f*rhovt(1))*lake_pow((xmas(1,ls)),(0.2635f)));
// WRF:7088
}
// WRF:7089
} else {
// WRF:7090
vtxbar(1,ls,2) = vtxbar(1,ls,1);
// WRF:7091
}
// WRF:7092
if((infdo>=2)) {
// WRF:7093
if((isnowfall==1)) {
// WRF:7094
vtxbar(1,ls,3) = ((6.12217f*rhovt(1))*lake_pow((xv(1,ls)),(lake_div(1.f,12.f))));
// WRF:7095
} else if((isnowfall==2)) {
// WRF:7096
vtxbar(1,ls,3) = ((13.3436f*rhovt(1))*lake_pow((xv(1,ls)),(0.14f)));
// WRF:7097
} else if((isnowfall==3)) {
// WRF:7099
vtxbar(1,ls,3) = ((61.0914f*rhovt(1))*lake_pow((xmas(1,ls)),(0.2635f)));
// WRF:7100
}
// WRF:7101
}
// WRF:7103
if(((sssflg<0)&&(temcg(1)>lake_abs(sssflg)))) {
// WRF:7104
vtxbar(1,ls,2) = vtxbar(1,ls,1);
// WRF:7105
vtxbar(1,ls,3) = vtxbar(1,ls,1);
// WRF:7106
}
// WRF:7108
}
// WRF:7109
} else {
// WRF:7110
vtxbar(1,ls,1) = (((lake_div((cs*gf4ds),6.0f))*(lake_pow(xdia(1,ls,1),ds)))*rhovt(1));
// WRF:7111
vtxbar(1,ls,2) = vtxbar(1,ls,1);
// WRF:7112
}
// WRF:7113
} else {
// WRF:7114
vtxbar(1,ls,1) = 0.0f;
// WRF:7115
}
// WRF:7117
if((snowfallfac!=1.0f)) {
// WRF:7118
vtxbar(1,ls,1) = (snowfallfac*vtxbar(1,ls,1));
// WRF:7119
vtxbar(1,ls,2) = (snowfallfac*vtxbar(1,ls,2));
// WRF:7120
vtxbar(1,ls,3) = (snowfallfac*vtxbar(1,ls,3));
// WRF:7121
}
// WRF:7124
}
// WRF:7125
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7127
}
// WRF:7134
if(((lh>1)&&(((ildo==0)||(ildo==lh))))) {
// WRF:7136
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7137
vtxbar(1,lh,1) = 0.0f;
// WRF:7138
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:7139
cd = coeff_cdx{}(lh);
// WRF:7140
if((icdx==1)) {
// WRF:7141
cd = coeff_cdx{}(lh);
// WRF:7142
} else if((icdx==2)) {
// WRF:7145
cd = lake_max(0.45f,lake_min(1.0f,(0.45f+lake_div((0.35f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(1,lh)))))),((800.f-500.f))))));
// WRF:7147
} else if((icdx==3)) {
// WRF:7149
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(1,lh)))))),((800.f-170.0f))))));
// WRF:7150
} else if((icdx==4)) {
// WRF:7151
cd = lake_max(cdhmin,lake_min(cdhmax,(cdhmin+lake_div((((cdhmax-cdhmin))*((cdhdnmax-lake_max(cdhdnmin,lake_min(cdhdnmax,xdn(1,lh)))))),((cdhdnmax-cdhdnmin))))));
// WRF:7153
} else if((icdx==5)) {
// WRF:7154
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(1,lh),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7155
} else if((icdx==6)) {
// WRF:7156
indxr = (((int)(lake_div(((xdn(1,lh)-50.f)),100.f)))+1);
// WRF:7157
indxr = lake_min(ngdnmm,lake_max(1,indxr));
// WRF:7160
delrho = lake_max(0.0f,(0.01f*((xdn(1,lh)-coeff_mmgraupvt{}(indxr,1)))));
// WRF:7161
if((indxr<ngdnmm)) {
// WRF:7163
axx(1,lh) = (coeff_mmgraupvt{}(indxr,2)+(delrho*((coeff_mmgraupvt{}((indxr+1),2)-coeff_mmgraupvt{}(indxr,2)))));
// WRF:7164
bxx(1,lh) = (coeff_mmgraupvt{}(indxr,3)+(delrho*((coeff_mmgraupvt{}((indxr+1),3)-coeff_mmgraupvt{}(indxr,3)))));
// WRF:7167
} else {
// WRF:7168
axx(1,lh) = coeff_mmgraupvt{}(indxr,2);
// WRF:7169
bxx(1,lh) = coeff_mmgraupvt{}(indxr,3);
// WRF:7170
}
// WRF:7172
aax = axx(1,lh);
// WRF:7173
bbx = bxx(1,lh);
// WRF:7175
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(1,lh)))))),((800.f-170.0f))))));
// WRF:7177
} else if((icdx<=0)) {
// WRF:7178
aax = coeff_ax{}(lh);
// WRF:7179
bbx = coeff_bx{}(lh);
// WRF:7180
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(1,lh)))))),((800.f-170.0f))))));
// WRF:7181
} else {
// WRF:7182
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(1,lh)))))),((800.f-170.0f))))));
// WRF:7183
}
// WRF:7185
cdxgs(1,lh) = cd;
// WRF:7186
if((((alpha(1,lh)==0.0f)&&(icdx>0))&&(icdx!=6))) {
// WRF:7190
axx(1,lh) = sqrtf(lake_div(((4.0f*xdn(1,lh))*gr),(((3.0f*cd)*rho00))));
// WRF:7191
bxx(1,lh) = 0.5f;
// WRF:7192
vtxbar(1,lh,1) = ((((lake_div(gf4p5,6.0f))*rhovt(1))*axx(1,lh))*sqrtf(xdia(1,lh,1)));
// WRF:7196
} else {
// WRF:7197
if((icdx!=6)) {bbx = coeff_bx{}(lh);}
// WRF:7198
tmp = ((4.f+alpha(1,lh))+bbx);
// WRF:7199
i = ((int)((dgami*(tmp))));
// WRF:7200
del = (tmp-(dgam*i));
// WRF:7201
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7203
tmp = (4.f+alpha(1,lh));
// WRF:7204
i = ((int)((dgami*(tmp))));
// WRF:7205
del = (tmp-(dgam*i));
// WRF:7206
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7211
if(((icdx>0)&&(icdx!=6))) {
// WRF:7212
aax = sqrtf(lake_div(((4.0f*xdn(1,lh))*gr),(((3.0f*cd)*rho00))));
// WRF:7213
vtxbar(1,lh,1) = lake_div((((rhovt(1)*aax)*sqrtf(xdia(1,lh,1)))*x),y);
// WRF:7214
axx(1,lh) = aax;
// WRF:7215
bxx(1,lh) = bbx;
// WRF:7216
} else if((icdx==6)) {
// WRF:7217
vtxbar(1,lh,1) = lake_div((((rhovt(1)*aax)*lake_pow(xdia(1,lh,1),bbx))*x),y);
// WRF:7218
} else {
// WRF:7219
axx(1,lh) = coeff_ax{}(lh);
// WRF:7220
bxx(1,lh) = coeff_bx{}(lh);
// WRF:7221
vtxbar(1,lh,1) = lake_div(((rhovt(1)*coeff_ax{}(lh))*((lake_pow(xdia(1,lh,1),coeff_bx{}(lh))*x))),y);
// WRF:7222
}
// WRF:7225
}
// WRF:7227
if((lwsm6&&(ipconc==0))) {
// WRF:7229
vtxbar(1,lh,1) = (((lake_div((330.f*gf4br),6.0f))*(lake_pow(xdia(1,lh,1),br)))*rhovt(1));
// WRF:7230
}
// WRF:7232
}
// WRF:7233
}
// WRF:7234
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7236
}
// WRF:7243
if(((lhl>1)&&(((ildo==0)||(ildo==lhl))))) {
// WRF:7245
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7246
vtxbar(1,lhl,1) = 0.0f;
// WRF:7247
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:7249
if((icdxhl==1)) {
// WRF:7250
cd = coeff_cdx{}(lhl);
// WRF:7251
} else if((icdxhl==3)) {
// WRF:7253
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(1,lhl)))))),((800.f-170.0f))))));
// WRF:7254
} else if((icdxhl==4)) {
// WRF:7255
cd = lake_max(cdhlmin,lake_min(cdhlmax,(cdhlmin+lake_div((((cdhlmax-cdhlmin))*((cdhldnmax-lake_max(cdhldnmin,lake_min(cdhldnmax,xdn(1,lhl)))))),((cdhldnmax-cdhldnmin))))));
// WRF:7257
} else if((icdxhl==5)) {
// WRF:7258
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(1,lhl),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7259
} else if((icdxhl==6)) {
// WRF:7260
indxr = (((int)(lake_div(((xdn(1,lhl)-50.f)),100.f)))+1);
// WRF:7261
indxr = lake_min(ngdnmm,lake_max(1,indxr));
// WRF:7264
delrho = lake_max(0.0f,(0.01f*((xdn(1,lhl)-coeff_mmgraupvt{}(indxr,1)))));
// WRF:7265
if((indxr<ngdnmm)) {
// WRF:7267
axx(1,lhl) = (coeff_mmgraupvt{}(indxr,2)+(delrho*((coeff_mmgraupvt{}((indxr+1),2)-coeff_mmgraupvt{}(indxr,2)))));
// WRF:7268
bxx(1,lhl) = (coeff_mmgraupvt{}(indxr,3)+(delrho*((coeff_mmgraupvt{}((indxr+1),3)-coeff_mmgraupvt{}(indxr,3)))));
// WRF:7271
} else {
// WRF:7272
axx(1,lhl) = coeff_mmgraupvt{}(indxr,2);
// WRF:7273
bxx(1,lhl) = coeff_mmgraupvt{}(indxr,3);
// WRF:7274
}
// WRF:7276
aax = axx(1,lhl);
// WRF:7277
bbx = bxx(1,lhl);
// WRF:7279
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(1,lhl)))))),((800.f-170.0f))))));
// WRF:7281
} else {
// WRF:7285
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(1,lhl)))))),((800.f-170.0f))))));
// WRF:7286
}
// WRF:7288
cdxgs(1,lhl) = cd;
// WRF:7290
if((((alpha(1,lhl)==0.0f)&&(icdxhl>0))&&(icdxhl!=6))) {
// WRF:7294
axx(1,lhl) = sqrtf(lake_div(((4.0f*xdn(1,lhl))*gr),(((3.0f*cd)*rho00))));
// WRF:7295
bxx(1,lhl) = 0.5f;
// WRF:7296
vtxbar(1,lhl,1) = ((((lake_div(gf4p5,6.0f))*rhovt(1))*axx(1,lhl))*sqrtf(xdia(1,lhl,1)));
// WRF:7297
} else {
// WRF:7298
if((icdxhl!=6)) {bbx = coeff_bx{}(lhl);}
// WRF:7299
tmp = ((4.f+alpha(1,lhl))+bbx);
// WRF:7300
i = ((int)((dgami*(tmp))));
// WRF:7301
del = (tmp-(dgam*i));
// WRF:7302
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7304
tmp = (4.f+alpha(1,lhl));
// WRF:7305
i = ((int)((dgami*(tmp))));
// WRF:7306
del = (tmp-(dgam*i));
// WRF:7307
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7309
if(((icdxhl>0)&&(icdxhl!=6))) {
// WRF:7310
aax = sqrtf(lake_div(((4.0f*xdn(1,lhl))*gr),(((3.0f*cd)*rho00))));
// WRF:7311
vtxbar(1,lhl,1) = lake_div((((rhovt(1)*aax)*sqrtf(xdia(1,lhl,1)))*x),y);
// WRF:7312
axx(1,lhl) = aax;
// WRF:7313
bxx(1,lhl) = bbx;
// WRF:7314
} else if((icdxhl==6)) {
// WRF:7315
vtxbar(1,lhl,1) = lake_div((((rhovt(1)*aax)*lake_pow((xdia(1,lhl,1)),bbx))*x),y);
// WRF:7316
} else {
// WRF:7317
axx(1,lhl) = coeff_ax{}(lhl);
// WRF:7318
bxx(1,lhl) = coeff_bx{}(lhl);
// WRF:7319
vtxbar(1,lhl,1) = lake_div((rhovt(1)*(((coeff_ax{}(lhl)*lake_pow(xdia(1,lhl,1),coeff_bx{}(lhl)))*x))),y);
// WRF:7320
}
// WRF:7323
}
// WRF:7326
}
// WRF:7327
}
// WRF:7328
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7330
}
// WRF:7333
if((infdo>=1)) {
// WRF:7337
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7338
if(((ildo==0)||(ildo==lc))) {
// WRF:7339
vtxbar(1,lc,2) = vtxbar(1,lc,1);
// WRF:7340
}
// WRF:7341
if((li>1)) {
// WRF:7352
}
// WRF:7354
}
// WRF:7356
if((lg>lr)) {
// WRF:7358
#pragma unroll
for(il=lg;il<=lhab;il+=1) {
// WRF:7359
if(((ildo==0)||(ildo==il))) {
// WRF:7361
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7362
if((qx(1,il)>coeff_qxmin{}(il))) {
// WRF:7363
if(((((il==lh)&&(hssflg==1)))||((((lhl>1)&&(il==lhl))&&(hlssflg==1))))) {
// WRF:7368
if((il==lh)) {
// WRF:7370
if((icdx==1)) {
// WRF:7371
cd = coeff_cdx{}(lh);
// WRF:7372
} else if((icdx==2)) {
// WRF:7375
cd = lake_max(0.45f,lake_min(1.0f,(0.45f+lake_div((0.35f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(1,lh)))))),((800.f-500.f))))));
// WRF:7377
} else if((icdx==3)) {
// WRF:7379
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hdnmn,lake_min(800.0f,xdn(1,lh)))))),((800.f-170.0f))))));
// WRF:7380
} else if((icdx==4)) {
// WRF:7381
cd = lake_max(cdhmin,lake_min(cdhmax,(cdhmin+lake_div((((cdhmax-cdhmin))*((cdhdnmax-lake_max(cdhdnmin,lake_min(cdhdnmax,xdn(1,lh)))))),((cdhdnmax-cdhdnmin))))));
// WRF:7383
} else if((icdx==5)) {
// WRF:7384
cd = (coeff_cdx{}(lh)*lake_pow((lake_div(xdn(1,lh),rho_qh)),(lake_div(2.f,3.f))));
// WRF:7385
} else if((icdx==6)) {
// WRF:7386
aax = axx(1,lh);
// WRF:7387
bbx = bxx(1,lh);
// WRF:7388
} else if((icdx<=0)) {
// WRF:7389
aax = coeff_ax{}(lh);
// WRF:7390
bbx = coeff_bx{}(lh);
// WRF:7391
}
// WRF:7393
} else if(((lhl>1)&&(il==lhl))) {
// WRF:7395
if((icdxhl==1)) {
// WRF:7396
cd = coeff_cdx{}(lhl);
// WRF:7397
} else if((icdxhl==3)) {
// WRF:7399
cd = lake_max(0.45f,lake_min(1.2f,(0.45f+lake_div((0.55f*((800.0f-lake_max(hldnmn,lake_min(800.0f,xdn(1,lhl)))))),((800.f-170.0f))))));
// WRF:7400
} else if((icdxhl==4)) {
// WRF:7401
cd = lake_max(cdhlmin,lake_min(cdhlmax,(cdhlmin+lake_div((((cdhlmax-cdhlmin))*((cdhldnmax-lake_max(cdhldnmin,lake_min(cdhldnmax,xdn(1,lhl)))))),((cdhldnmax-cdhldnmin))))));
// WRF:7403
} else if((icdxhl==5)) {
// WRF:7406
cd = lake_max(0.45f,lake_min(0.6f,(0.45f+lake_div((0.15f*((800.0f-lake_max(500.f,lake_min(800.0f,xdn(1,lhl)))))),((800.f-500.f))))));
// WRF:7407
} else if((icdxhl==6)) {
// WRF:7408
aax = axx(1,lhl);
// WRF:7409
bbx = bxx(1,lhl);
// WRF:7410
} else if((icdxhl<=0)) {
// WRF:7411
aax = coeff_ax{}(lhl);
// WRF:7412
bbx = coeff_bx{}(lhl);
// WRF:7413
}
// WRF:7415
}
// WRF:7417
if((((alpha(1,il)==0.f)&&(infdo<2))&&((((((il==lh)&&(icdx>0))&&(icdx!=6)))||((((il==lhl)&&(icdxhl>0))&&(icdxhl!=6))))))) {
// WRF:7419
vtxbar(1,il,2) = sqrtf(lake_div(((((xdn(1,il)*xdia(1,il,1))*pi)*gr)),(((3.0f*cd)*lake_max(0.05f,rho0(1))))));
// WRF:7423
} else {
// WRF:7424
if(((il==lh)&&(icdx!=6))) {bbx = coeff_bx{}(il);}
// WRF:7425
if(((il==lhl)&&(icdxhl!=6))) {bbx = coeff_bx{}(il);}
// WRF:7426
tmp = ((1.f+alpha(1,il))+bbx);
// WRF:7427
i = ((int)((dgami*(tmp))));
// WRF:7428
del = (tmp-(dgam*i));
// WRF:7429
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7431
tmp = (1.f+alpha(1,il));
// WRF:7432
i = ((int)((dgami*(tmp))));
// WRF:7433
del = (tmp-(dgam*i));
// WRF:7434
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7436
if(((il==lh)||(il==lhl))) {
// WRF:7437
if((((il==lh)&&(icdx>0)))) {
// WRF:7438
if((icdx!=6)) {
// WRF:7439
aax = sqrtf(lake_div(((4.0f*xdn(1,il))*gr),(((3.0f*cd)*rho00))));
// WRF:7440
vtxbar(1,il,2) = lake_div((((rhovt(1)*aax)*lake_pow(xdia(1,il,1),coeff_bx{}(il)))*x),y);
// WRF:7441
} else {
// WRF:7442
vtxbar(1,il,2) = lake_div((((rhovt(1)*aax)*lake_pow(xdia(1,il,1),bbx))*x),y);
// WRF:7443
}
// WRF:7445
} else if((((il==lhl)&&(icdxhl>0)))) {
// WRF:7446
if((icdxhl!=6)) {
// WRF:7447
aax = sqrtf(lake_div(((4.0f*xdn(1,il))*gr),(((3.0f*cd)*rho00))));
// WRF:7448
vtxbar(1,il,2) = lake_div((((rhovt(1)*aax)*lake_pow(xdia(1,il,1),coeff_bx{}(il)))*x),y);
// WRF:7449
} else {
// WRF:7450
vtxbar(1,il,2) = lake_div((((rhovt(1)*aax)*lake_pow(xdia(1,il,1),bbx))*x),y);
// WRF:7451
}
// WRF:7452
} else {
// WRF:7453
aax = coeff_ax{}(il);
// WRF:7454
vtxbar(1,il,2) = lake_div(((rhovt(1)*coeff_ax{}(il))*((lake_pow(xdia(1,il,1),coeff_bx{}(il))*x))),y);
// WRF:7455
}
// WRF:7462
if((infdo>=2)) {
// WRF:7464
tmp = ((7.f+alpha(1,il))+bbx);
// WRF:7465
i = ((int)((dgami*(tmp))));
// WRF:7466
del = (tmp-(dgam*i));
// WRF:7467
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7469
tmp = (7.f+alpha(1,il));
// WRF:7470
i = ((int)((dgami*(tmp))));
// WRF:7471
del = (tmp-(dgam*i));
// WRF:7472
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:7474
vtxbar(1,il,3) = lake_div((rhovt(1)*(((aax*lake_pow((xdia(1,il,1)),bbx))*x))),y);
// WRF:7478
if(((!(((vtxbar(1,il,1)>(-1.f))&&(vtxbar(1,il,1)<200.f))))||(!(((vtxbar(1,il,3)>(-1.f))&&(vtxbar(1,il,3)<200.f)))))) {
// WRF:7480
; /* native diagnostic only */
// WRF:7481
; /* native diagnostic only */
// WRF:7483
}
// WRF:7486
}
// WRF:7488
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7490
} else {
// WRF:7491
vtxbar(1,il,2) = lake_div((rhovt(1)*(((coeff_ax{}(il)*lake_pow(xdia(1,il,1),coeff_bx{}(il)))*x))),y);
// WRF:7495
if((infdo>=2)) {
// WRF:7496
vtxbar(1,il,3) = lake_div((rhovt(1)*(((aax*lake_pow((lake_div(1.0f,xdia(1,il,1))),((-bbx))))*gamma_sp(((7.0f+alpha(1,il))+bbx))))),gamma_sp((7.f+alpha(1,il))));
// WRF:7501
}
// WRF:7503
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7505
}
// WRF:7507
}
// WRF:7518
} else if(((((il==lh)&&(hssflg==0)))||((((lhl>1)&&(il==lhl))&&(hlssflg==0))))) {
// WRF:7519
vtxbar(1,il,2) = vtxbar(1,il,1);
// WRF:7520
vtxbar(1,il,3) = vtxbar(1,il,1);
// WRF:7521
} else {
// WRF:7522
vtxbar(1,il,2) = sqrtf(lake_div(((((xdn(1,il)*xdia(1,il,1))*pi)*gr)),(((3.0f*coeff_cdx{}(il))*lake_max(0.05f,rho0(1))))));
// WRF:7525
vtxbar(1,il,3) = vtxbar(1,il,1);
// WRF:7527
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7530
}
// WRF:7531
} else {
// WRF:7532
vtxbar(1,il,2) = 0.0f;
// WRF:7534
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7536
}
// WRF:7537
}
// WRF:7539
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7541
}
// WRF:7542
}
// WRF:7544
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7546
}
// WRF:7551
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7560
}
// WRF:7562
if(((lh>0)&&(graupelfallfac!=1.0f))) {
// WRF:7563
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7564
vtxbar(1,lh,1) = (graupelfallfac*vtxbar(1,lh,1));
// WRF:7565
vtxbar(1,lh,2) = (graupelfallfac*vtxbar(1,lh,2));
// WRF:7566
vtxbar(1,lh,3) = (graupelfallfac*vtxbar(1,lh,3));
// WRF:7567
axx(1,lh) = (graupelfallfac*axx(1,lh));
// WRF:7568
}
// WRF:7569
}
// WRF:7571
if(((lhl>0)&&(hailfallfac!=1.0f))) {
// WRF:7572
for(int _point_once=0;_point_once<(ngscnt>0 ? 1 : 0);++_point_once) {
// WRF:7573
vtxbar(1,lhl,1) = (hailfallfac*vtxbar(1,lhl,1));
// WRF:7574
vtxbar(1,lhl,2) = (hailfallfac*vtxbar(1,lhl,2));
// WRF:7575
vtxbar(1,lhl,3) = (hailfallfac*vtxbar(1,lhl,3));
// WRF:7576
axx(1,lhl) = (hailfallfac*axx(1,lhl));
// WRF:7577
}
// WRF:7578
}
// WRF:7580
if((ndebug1>0)) {; /* native diagnostic only */}
// WRF:7584
return;
}
__device__ __forceinline__ void ziegfall1d(int nx,int ny,int nz,int nor,int norz,int na,float dtp,int jgs,int ixcol,Array<float> xvt,Array<const float> rhovtzx,Array<float> an,Array<const float> dn,int ipconc0,Array<const float> t0,Array<const float> t7,float cwmasn,float cwmasx,float cwradn,Array<const float> qxmin,Array<const float> xdnmx,Array<const float> xdnmn,Array<const float> cdx,Array<const float> cno,Array<const float> xdn0,Array<const float> xvmn,Array<const float> xvmx,int ngs,Array<float> qx,Array<const float> qxw,Array<float> cx,Array<float> xv,Array<float> vtxbar,Array<float> xmas,Array<float> xdn,Array<float> xdia,Array<float> vx,Array<float> alpha,Array<float> zx,Array<int> igs,Array<int> kgs,Array<float> rho0,Array<float> temcg,Array<float> temg,Array<float> rhovt,Array<const float> cwnc,Array<const float> cinc,Array<float> fadvisc,Array<const float> cwdia,Array<const float> cipmas,Array<float> cnina,Array<const float> cimas,Array<float> cnostmp,int infdo,int ildo,double timesetvt=0) {
constexpr int ng1=1;
int ngt;
float dtz1;
constexpr int ndebugzf=0;
int ix;
int jy;
int kz;
int i;
int j;
int k;
int il;
int ngscnt;
int mgs;
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> cdxgs{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lh+1),1,(NZ+3),lh,lhab> axx{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lh+1),1,(NZ+3),lh,lhab> bxx{};
float cwccn0;
int nxmpb;
int nzmpb;
int nxz;
int numgs;
int inumgs;
constexpr int kstag=1;
bool flag;
bool ldoliq;
float chw;
float qr;
float z;
float rd;
float alp;
float z1;
float g1;
float vr;
float nrx;
float tmp;
float vtmax;
float xvbarmax;
const float c1r=19.0f;
const float c2r=0.6f;
const float c3r=1.8f;
const float c4r=17.0f;
const float c1h=5.5f;
const float c2h=0.7f;
const float c3h=4.5f;
const float c4h=8.5f;
const float c1hl=3.7f;
const float c2hl=0.3f;
const float c3hl=9.0f;
const float c4hl=6.5f;
const float c5hl=1.0f;
const float c6hl=6.5f;
int l1;
int l2;
double dpt1;
double dpt2;
int ixb;
int jyb;
int kzb;
int ixe;
int jye;
int kze;
bool debug_mpi=false;
// WRF:7738
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:7748
ldoliq = false;
// WRF:7749
if((ls>1)) {
// WRF:7750
#pragma unroll
for(il=ls;il<=lhab;il+=1) {
// WRF:7751
ldoliq = (ldoliq||((coeff_lliq{}(il)>1)));
// WRF:7752
}
// WRF:7753
}
// WRF:7787
jy = jgs;
// WRF:7788
nxmpb = ixcol;
// WRF:7789
nzmpb = 1;
// WRF:7790
nxz = (1*nz);
// WRF:7792
numgs = 1;
// WRF:7794
if((ildo==0)) {
// WRF:7795
l1 = lc;
// WRF:7796
l2 = lhab;
// WRF:7797
} else {
// WRF:7798
l1 = ildo;
// WRF:7799
l2 = ildo;
// WRF:7800
}
// WRF:7803
for(inumgs=1;inumgs<=numgs;inumgs+=1) {
// WRF:7804
ngscnt = 0;
// WRF:7807
for(kz=nzmpb;kz<=nz;kz+=1) {
// WRF:7808
for(ix=ixcol;ix<=ixcol;ix+=1) {
// WRF:7809
flag = false;
// WRF:7812
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:7813
flag = (flag||((an(ix,jy,kz,il)>coeff_qxmin{}(il))));
// WRF:7814
}
// WRF:7816
if(flag) {
// WRF:7819
ngscnt = (ngscnt+1);
// WRF:7820
igs(ngscnt) = ix;
// WRF:7821
kgs(ngscnt) = kz;
// WRF:7822
if((ngscnt==ngs)) {goto L1100;}
// WRF:7823
}
// WRF:7824
}
// WRF:7825
nxmpb = 1;
// WRF:7826
}
// WRF:7830
L1100: ;
;
// WRF:7832
if((ngscnt==0)) {goto L9998;}
// WRF:7841
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7843
rho0(mgs) = dn(igs(mgs),jy,kgs(mgs));
// WRF:7844
rhovt(mgs) = rhovtzx(kgs(mgs),ixcol);
// WRF:7845
temg(mgs) = t0(igs(mgs),jy,kgs(mgs));
// WRF:7846
temcg(mgs) = (temg(mgs)-tfr);
// WRF:7850
}
// WRF:7853
if(((lc>1)&&(((ildo==0)||(ildo==lc))))) {
// WRF:7854
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7855
fadvisc(mgs) = ((advisc0*(lake_div(416.16f,((temg(mgs)+120.0f)))))*lake_pow((lake_div(temg(mgs),296.0f)),(1.5f)));
// WRF:7857
}
// WRF:7858
}
// WRF:7860
if((ipconc==0)) {
// WRF:7861
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7862
cnina(mgs) = t7(igs(mgs),jgs,kgs(mgs));
// WRF:7863
}
// WRF:7864
}
// WRF:7867
if((ildo>0)) {
// WRF:7868
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s2=1;_s2<=3;++_s2) {vtxbar(_s0,ildo,_s2) = 0.0f;}}
// WRF:7869
} else {
// WRF:7870
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {for(int _s2=1;_s2<=3;++_s2) {vtxbar(_s0,_s1,_s2) = 0.0f;}}}
// WRF:7871
}
// WRF:7876
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:7877
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7878
qx(mgs,il) = lake_max(an(igs(mgs),jy,kgs(mgs),il),0.0f);
// WRF:7879
}
// WRF:7880
}
// WRF:7882
for(int _s0=1;_s0<=ngs;++_s0) {cnostmp(_s0) = coeff_cno{}(ls);}
// WRF:7883
if((((ipconc<1)&&lwsm6)&&(((ildo==0)||(ildo==ls))))) {
// WRF:7884
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7885
tmp = lake_min(0.0f,temcg(mgs));
// WRF:7886
cnostmp(mgs) = lake_min(2.e8f,(2.e6f*gfk_exp((0.12f*tmp))));
// WRF:7887
}
// WRF:7888
}
// WRF:7894
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {cx(_s0,_s1) = 0.0f;}}
// WRF:7896
if((((ipconc>=1)&&(li>1))&&(((ildo==0)||(ildo==li))))) {
// WRF:7897
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7898
cx(mgs,li) = lake_max(an(igs(mgs),jy,kgs(mgs),lni),0.0f);
// WRF:7899
}
// WRF:7900
}
// WRF:7901
if((((ipconc>=2)&&(lc>1))&&(((ildo==0)||(ildo==lc))))) {
// WRF:7902
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7903
cx(mgs,lc) = lake_max(an(igs(mgs),jy,kgs(mgs),lnc),0.0f);
// WRF:7905
}
// WRF:7906
}
// WRF:7907
if((((ipconc>=3)&&(lr>1))&&(((ildo==0)||(ildo==lr))))) {
// WRF:7908
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7909
cx(mgs,lr) = lake_max(an(igs(mgs),jy,kgs(mgs),lnr),0.0f);
// WRF:7914
}
// WRF:7915
}
// WRF:7916
if((((ipconc>=4)&&(ls>1))&&(((ildo==0)||(ildo==ls))))) {
// WRF:7917
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7918
cx(mgs,ls) = lake_max(an(igs(mgs),jy,kgs(mgs),lns),0.0f);
// WRF:7923
}
// WRF:7924
}
// WRF:7926
if((((ipconc>=5)&&(lh>1))&&(((ildo==0)||(ildo==lh))))) {
// WRF:7927
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7929
cx(mgs,lh) = lake_max(an(igs(mgs),jy,kgs(mgs),lnh),0.0f);
// WRF:7935
}
// WRF:7936
}
// WRF:7938
if((((ipconc>=5)&&(lhl>1))&&(((ildo==0)||(ildo==lhl))))) {
// WRF:7939
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7941
cx(mgs,lhl) = lake_max(an(igs(mgs),jy,kgs(mgs),lnhl),0.0f);
// WRF:7950
}
// WRF:7951
}
// WRF:7953
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7954
xdn(mgs,lc) = coeff_xdn0{}(lc);
// WRF:7955
xdn(mgs,lr) = coeff_xdn0{}(lr);
// WRF:7958
if((li>1)) {xdn(mgs,li) = coeff_xdn0{}(li);}
// WRF:7959
if((ls>1)) {xdn(mgs,ls) = coeff_xdn0{}(ls);}
// WRF:7960
if((lh>1)) {xdn(mgs,lh) = coeff_xdn0{}(lh);}
// WRF:7961
if((lhl>1)) {xdn(mgs,lhl) = coeff_xdn0{}(lhl);}
// WRF:7962
}
// WRF:7967
if((ldovol&&(((ildo==0)||(ildo>=li))))) {
// WRF:7969
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=li;_s1<=lhab;++_s1) {vx(_s0,_s1) = 0.0f;}}
// WRF:7971
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:7973
if((coeff_lvol{}(il)>=1)) {
// WRF:7975
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7976
vx(mgs,il) = lake_max(an(igs(mgs),jy,kgs(mgs),coeff_lvol{}(il)),0.0f);
// WRF:7977
if(((vx(mgs,il)>((rho0(mgs)*coeff_qxmin{}(il))*1.e-3f))&&(qx(mgs,il)>coeff_qxmin{}(il)))) {
// WRF:7978
xdn(mgs,il) = lake_min(coeff_xdnmx{}(il),lake_max(coeff_xdnmn{}(il),lake_div((rho0(mgs)*qx(mgs,il)),vx(mgs,il))));
// WRF:7979
}
// WRF:7980
}
// WRF:7982
}
// WRF:7984
}
// WRF:7986
}
// WRF:7988
#pragma unroll
for(il=lg;il<=lhab;il+=1) {
// WRF:7989
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:7990
alpha(mgs,il) = coeff_dnu{}(il);
// WRF:7991
}
// WRF:7992
}
// WRF:7994
if((imurain==1)) {
// WRF:7995
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = alphar;}
// WRF:7996
} else if((imurain==3)) {
// WRF:7997
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = coeff_xnu{}(lr);}
// WRF:7998
}
// WRF:8001
if(((ipconc==5)&&(imydiagalpha>0))) {
// WRF:8002
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:8003
if(((qx(mgs,lr)>coeff_qxmin{}(lr))&&(cx(mgs,lr)>cxmin))) {
// WRF:8004
xv(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((xdn(mgs,lr)*cx(mgs,lr))));
// WRF:8005
xdia(mgs,lr,3) = lake_pow((((xv(mgs,lr)*6.0f)*cwc1)),(lake_div(1.f,3.f)));
// WRF:8006
alpha(mgs,lr) = lake_min(alphamax,((c1r*tanhf((c2r*(((xdia(mgs,lr,3)*1000.f)-c3r)))))+c4r));
// WRF:8007
}
// WRF:8008
if(((qx(mgs,lh)>coeff_qxmin{}(lh))&&(cx(mgs,lh)>cxmin))) {
// WRF:8009
xv(mgs,lh) = lake_div((rho0(mgs)*qx(mgs,lh)),((xdn(mgs,lh)*cx(mgs,lh))));
// WRF:8010
xdia(mgs,lh,3) = lake_pow((((xv(mgs,lh)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:8011
alpha(mgs,lh) = lake_min(alphamax,((c1h*tanhf((c2h*(((xdia(mgs,lh,3)*1000.f)-c3h)))))+c4h));
// WRF:8012
}
// WRF:8015
if((lhl>0)) {
// WRF:8016
if(((qx(mgs,lhl)>coeff_qxmin{}(lhl))&&(cx(mgs,lhl)>cxmin))) {
// WRF:8017
xv(mgs,lhl) = lake_div((rho0(mgs)*qx(mgs,lhl)),((xdn(mgs,lhl)*cx(mgs,lhl))));
// WRF:8018
xdia(mgs,lhl,3) = lake_pow((((xv(mgs,lhl)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:8019
if((xdia(mgs,lhl,3)<0.008f)) {
// WRF:8020
alpha(mgs,lhl) = lake_min(alphamax,((c1hl*tanhf((c2hl*(((xdia(mgs,lhl,3)*1000.f)-c3hl)))))+c4hl));
// WRF:8021
} else {
// WRF:8022
alpha(mgs,lhl) = lake_min(alphamax,(((c5hl*xdia(mgs,lhl,3))*1000.f)+c6hl));
// WRF:8023
}
// WRF:8024
}
// WRF:8025
}
// WRF:8026
}
// WRF:8027
}
// WRF:8033
if(((ipconc>=6)||(lzr>1))) {
// WRF:8035
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lr;_s1<=lhab;++_s1) {zx(_s0,_s1) = 0.0f;}}
// WRF:8038
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:8040
if((coeff_lz{}(il)>=1)) {
// WRF:8042
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:8043
zx(mgs,il) = lake_max(an(igs(mgs),jy,kgs(mgs),coeff_lz{}(il)),0.0f);
// WRF:8044
}
// WRF:8047
}
// WRF:8049
}
// WRF:8051
}
// WRF:8060
if((((coeff_lz{}(lr)>1)&&(((ildo==0)||(ildo==lr))))&&(imurain==3))) {
// WRF:8061
il = lr;
// WRF:8062
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:8064
if(((iresetmoments==1)||(iresetmoments==il))) {
// WRF:8066
if((zx(mgs,lr)<=zxmin)) {
// WRF:8067
qx(mgs,lr) = 0.0f;
// WRF:8068
cx(mgs,lr) = 0.0f;
// WRF:8069
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),lr));
// WRF:8070
an(igs(mgs),jgs,kgs(mgs),lr) = qx(mgs,lr);
// WRF:8071
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(lr)) = cx(mgs,lr);
// WRF:8074
} else if((cx(mgs,lr)<=cxmin)) {
// WRF:8075
zx(mgs,lr) = 0.0f;
// WRF:8076
qx(mgs,lr) = 0.0f;
// WRF:8077
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),lr));
// WRF:8078
an(igs(mgs),jgs,kgs(mgs),lr) = qx(mgs,lr);
// WRF:8079
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(lr)) = zx(mgs,lr);
// WRF:8080
}
// WRF:8081
}
// WRF:8085
if((qx(mgs,lr)>coeff_qxmin{}(lr))) {
// WRF:8087
xv(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((xdn(mgs,lr)*lake_max(1.0e-11f,cx(mgs,lr)))));
// WRF:8088
if((xv(mgs,lr)>coeff_xvmx{}(lr))) {
// WRF:8098
} else if((xv(mgs,lr)<coeff_xvmn{}(lr))) {
// WRF:8099
xv(mgs,lr) = coeff_xvmn{}(lr);
// WRF:8100
cx(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((coeff_xvmn{}(lr)*xdn(mgs,lr))));
// WRF:8101
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8102
}
// WRF:8104
if(((zx(mgs,il)>0.0f)&&(cx(mgs,il)<=0.0f))) {
// WRF:8106
g1 = lake_div((36.f*((alpha(mgs,lr)+2.0f))),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8107
z = zx(mgs,il);
// WRF:8108
qr = qx(mgs,il);
// WRF:8110
cx(mgs,il) = lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:8111
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8113
} else if(((zx(mgs,il)<=zxmin)&&(cx(mgs,il)>cxmin))) {
// WRF:8115
g1 = lake_div((36.f*((alpha(mgs,lr)+2.0f))),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8116
chw = cx(mgs,il);
// WRF:8117
qr = qx(mgs,il);
// WRF:8126
zx(mgs,il) = lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),((lake_pow(xdn(mgs,lr),2)*chw)));
// WRF:8127
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8129
} else if(((zx(mgs,il)<=zxmin)&&(cx(mgs,il)<=0.0f))) {
// WRF:8133
if((qx(mgs,il)<1.e-8f)) {
// WRF:8134
qx(mgs,il) = 0.0f;
// WRF:8135
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8136
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8137
} else {
// WRF:8139
zx(mgs,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:8140
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8142
g1 = lake_div((36.f*((alpha(mgs,lr)+2.0f))),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8143
z = zx(mgs,il);
// WRF:8144
qr = qx(mgs,il);
// WRF:8145
cx(mgs,il) = lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:8146
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8147
}
// WRF:8148
}
// WRF:8150
if((zx(mgs,lr)>0.0f)) {
// WRF:8151
xv(mgs,lr) = lake_div((rho0(mgs)*qx(mgs,lr)),((1000.f*lake_max(1.0e-9f,cx(mgs,lr)))));
// WRF:8152
vr = xv(mgs,lr);
// WRF:8154
qr = qx(mgs,lr);
// WRF:8155
nrx = cx(mgs,lr);
// WRF:8156
z = zx(mgs,lr);
// WRF:8162
if((z>0.0f)) {
// WRF:8164
alp = (lake_div((((36.f*((alpha(mgs,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8165
for(i=1;i<=20;i+=1) {
// WRF:8167
if((lake_abs((alp-alpha(mgs,lr)))<0.01f)) {break;}
// WRF:8168
alpha(mgs,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8169
alp = (lake_div((((36.f*((alpha(mgs,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8171
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8172
}
// WRF:8177
if((xv(mgs,il)>coeff_xvmx{}(il))) {
// WRF:8178
tmp = cx(mgs,il);
// WRF:8179
xv(mgs,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(mgs,il)));
// WRF:8180
xmas(mgs,il) = (xv(mgs,il)*xdn(mgs,il));
// WRF:8181
cx(mgs,il) = lake_div((rho0(mgs)*qx(mgs,il)),(xmas(mgs,il)));
// WRF:8182
if((tmp<cx(mgs,il))) {
// WRF:8184
g1 = lake_div((36.f*((alpha(mgs,lr)+2.0f))),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8185
zx(mgs,il) = (zx(mgs,il)+((g1*lake_pow((lake_div(rho0(mgs),xdn(mgs,il))),2))*((lake_pow((lake_div(qx(mgs,il),tmp)),2)*((tmp-cx(mgs,il)))))));
// WRF:8186
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8188
vr = xv(mgs,lr);
// WRF:8189
qr = qx(mgs,lr);
// WRF:8190
nrx = cx(mgs,lr);
// WRF:8191
z = zx(mgs,lr);
// WRF:8195
alp = (lake_div((((36.f*((alpha(mgs,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8196
for(i=1;i<=20;i+=1) {
// WRF:8197
if((lake_abs((alp-alpha(mgs,lr)))<0.01f)) {break;}
// WRF:8198
alpha(mgs,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8199
alp = (lake_div((((36.f*((alpha(mgs,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8200
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8201
}
// WRF:8204
}
// WRF:8205
}
// WRF:8212
if((true&&((((alpha(mgs,il)<=rnumin)||(alp==rnumin))||(alp==rnumax))))) {
// WRF:8214
if((rescale_high_alpha&&(alp>=(rnumax-0.01f)))) {
// WRF:8215
g1 = lake_div((36.f*((alpha(mgs,lr)+2.0f))),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8216
cx(mgs,il) = (lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),z)*lake_pow((lake_div(1.f,(xdn(mgs,il)))),2));
// WRF:8217
an(igs(mgs),jy,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8219
} else if((rescale_low_alphar&&(alp<=rnumin))) {
// WRF:8221
z = lake_div((((36.f*((alpha(mgs,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((((alpha(mgs,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8222
zx(mgs,il) = z;
// WRF:8223
an(igs(mgs),jy,kgs(mgs),coeff_lz{}(il)) = z;
// WRF:8225
}
// WRF:8226
}
// WRF:8228
}
// WRF:8229
}
// WRF:8231
} else {
// WRF:8233
zx(mgs,lr) = 0.0f;
// WRF:8234
cx(mgs,lr) = 0.0f;
// WRF:8235
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(lr)) = cx(mgs,lr);
// WRF:8236
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(lr)) = zx(mgs,lr);
// WRF:8238
}
// WRF:8240
}
// WRF:8241
}
// WRF:8244
if((ipconc>=6)) {
// WRF:8248
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:8250
if((((coeff_lz{}(il)>1)&&(((ildo==0)||(ildo==il))))&&((!(((il==lr)&&(imurain==3))))))) {
// WRF:8252
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:8254
if((((iresetmoments==1)||(iresetmoments==il))||(iresetmoments==(-1)))) {
// WRF:8255
if((zx(mgs,il)<=zxmin)) {
// WRF:8256
qx(mgs,il) = 0.0f;
// WRF:8257
cx(mgs,il) = 0.0f;
// WRF:8258
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8259
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8260
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8261
} else if(((iresetmoments==(-1))&&(qx(mgs,il)<coeff_qxmin{}(il)))) {
// WRF:8262
zx(mgs,il) = 0.0f;
// WRF:8263
cx(mgs,il) = 0.0f;
// WRF:8264
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8266
qx(mgs,il) = 0.0f;
// WRF:8267
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8268
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8269
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8271
} else if(((cx(mgs,il)<=cxmin)&&(iresetmoments!=(-1)))) {
// WRF:8273
zx(mgs,il) = 0.0f;
// WRF:8274
qx(mgs,il) = 0.0f;
// WRF:8275
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8276
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8277
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8278
}
// WRF:8279
}
// WRF:8281
if(((zx(mgs,il)<=zxmin)&&(cx(mgs,il)<=cxmin))) {
// WRF:8282
zx(mgs,il) = 0.0f;
// WRF:8283
cx(mgs,il) = 0.0f;
// WRF:8284
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8285
qx(mgs,il) = 0.0f;
// WRF:8286
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8287
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8288
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8289
}
// WRF:8291
if((qx(mgs,il)>coeff_qxmin{}(il))) {
// WRF:8293
xv(mgs,il) = lake_div((rho0(mgs)*qx(mgs,il)),((xdn(mgs,il)*lake_max(1.0e-9f,cx(mgs,il)))));
// WRF:8294
xmas(mgs,il) = (xv(mgs,il)*xdn(mgs,il));
// WRF:8296
if((xv(mgs,il)<coeff_xvmn{}(il))) {
// WRF:8298
xv(mgs,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(mgs,il)));
// WRF:8299
xmas(mgs,il) = (xv(mgs,il)*xdn(mgs,il));
// WRF:8300
cx(mgs,il) = lake_div((rho0(mgs)*qx(mgs,il)),(xmas(mgs,il)));
// WRF:8308
}
// WRF:8310
if(((zx(mgs,il)>0.0f)&&(cx(mgs,il)<=0.0f))) {
// WRF:8312
g1 = lake_div(((((6.0f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*((1.0f+alpha(mgs,il))))));
// WRF:8314
z = zx(mgs,il);
// WRF:8315
qr = qx(mgs,il);
// WRF:8316
cx(mgs,il) = lake_div(((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*lake_pow(((6*qr)),2)),((z*lake_pow(((pi*xdn(mgs,il))),2))));
// WRF:8317
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8319
} else if(((zx(mgs,il)<=zxmin)&&(cx(mgs,il)>cxmin))) {
// WRF:8321
g1 = lake_div(((((6.0f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*((1.0f+alpha(mgs,il))))));
// WRF:8323
chw = cx(mgs,il);
// WRF:8324
qr = qx(mgs,il);
// WRF:8326
zx(mgs,il) = lake_min((zxmin*1.1f),lake_div(((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*lake_pow(((6*qr)),2)),((chw*lake_pow(((pi*xdn(mgs,il))),2)))));
// WRF:8327
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8328
} else if(((zx(mgs,il)<=zxmin)&&(cx(mgs,il)<=0.0f))) {
// WRF:8334
if((qx(mgs,il)<1.e-8f)) {
// WRF:8335
qx(mgs,il) = 0.0f;
// WRF:8336
an(igs(mgs),jgs,kgs(mgs),lv) = (an(igs(mgs),jgs,kgs(mgs),lv)+an(igs(mgs),jgs,kgs(mgs),il));
// WRF:8337
an(igs(mgs),jgs,kgs(mgs),il) = qx(mgs,il);
// WRF:8338
} else {
// WRF:8342
zx(mgs,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:8343
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8345
g1 = lake_div(((((6.0f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*((1.0f+alpha(mgs,il))))));
// WRF:8347
z = zx(mgs,il);
// WRF:8348
qr = qx(mgs,il);
// WRF:8349
cx(mgs,il) = lake_div(((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*lake_pow(((6*qr)),2)),((z*lake_pow(((pi*xdn(mgs,il))),2))));
// WRF:8350
an(igs(mgs),jgs,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8351
}
// WRF:8352
}
// WRF:8353
}
// WRF:8355
if(((qx(mgs,il)>coeff_qxmin{}(il))&&(cx(mgs,il)>cxmin))) {
// WRF:8356
chw = cx(mgs,il);
// WRF:8357
qr = qx(mgs,il);
// WRF:8358
z = zx(mgs,il);
// WRF:8360
if((zx(mgs,il)>0.f)) {
// WRF:8363
rd = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(mgs,il))),2))*chw),(lake_pow(((dn(igs(mgs),jy,kgs(mgs))*qr)),2)));
// WRF:8365
alp = (lake_div(((((6.f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*rd)))-1.0f);
// WRF:8367
for(i=1;i<=10;i+=1) {
// WRF:8368
if((lake_abs((alp-alpha(mgs,il)))<0.01f)) {break;}
// WRF:8369
alpha(mgs,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8370
alp = (lake_div(((((6.f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*rd)))-1.0f);
// WRF:8373
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8374
}
// WRF:8380
if((imaxdiaopt==1)) {
// WRF:8381
xvbarmax = coeff_xvmx{}(il);
// WRF:8382
} else if((imaxdiaopt==2)) {
// WRF:8383
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((3.f+alpha(mgs,il))),3),(((((3.f+alpha(mgs,il)))*((2.f+alpha(mgs,il))))*((1.f+alpha(mgs,il))))))));
// WRF:8384
} else if((imaxdiaopt==3)) {
// WRF:8385
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((4.f+alpha(mgs,il))),3),(((((3.f+alpha(mgs,il)))*((2.f+alpha(mgs,il))))*((1.f+alpha(mgs,il))))))));
// WRF:8386
}
// WRF:8388
if((xv(mgs,il)>xvbarmax)) {
// WRF:8389
tmp = cx(mgs,il);
// WRF:8390
xv(mgs,il) = lake_min(xvbarmax,lake_max(coeff_xvmn{}(il),xv(mgs,il)));
// WRF:8391
xmas(mgs,il) = (xv(mgs,il)*xdn(mgs,il));
// WRF:8392
cx(mgs,il) = lake_div((rho0(mgs)*qx(mgs,il)),(xmas(mgs,il)));
// WRF:8393
if((tmp<cx(mgs,il))) {
// WRF:8394
g1 = lake_div((((36.f*((6.0f+alpha(mgs,il))))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),((((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*((1.0f+alpha(mgs,il))))*lake_pow(pi,2))));
// WRF:8396
zx(mgs,il) = (zx(mgs,il)+((g1*lake_pow((lake_div(rho0(mgs),xdn(mgs,il))),2))*((lake_pow((lake_div(qx(mgs,il),tmp)),2)*((tmp-cx(mgs,il)))))));
// WRF:8397
an(igs(mgs),jgs,kgs(mgs),coeff_lz{}(il)) = zx(mgs,il);
// WRF:8399
chw = cx(mgs,il);
// WRF:8400
qr = qx(mgs,il);
// WRF:8401
z = zx(mgs,il);
// WRF:8403
rd = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(mgs,il))),2))*chw),(lake_pow(((rho0(mgs)*qr)),2)));
// WRF:8404
alp = (lake_div(((((6.0f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*rd)))-1.0f);
// WRF:8406
for(i=1;i<=10;i+=1) {
// WRF:8407
if((lake_abs((alp-alpha(mgs,il)))<0.01f)) {break;}
// WRF:8408
alpha(mgs,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8409
alp = (lake_div(((((6.f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*rd)))-1.0f);
// WRF:8411
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8412
}
// WRF:8415
}
// WRF:8416
}
// WRF:8422
if((((rescale_low_alpha||rescale_high_alpha))&&((((alpha(mgs,il)<=alphamin)||(alp==alphamin))||(alp==alphamax))))) {
// WRF:8425
g1 = lake_div(((((6.0f+alpha(mgs,il)))*((5.0f+alpha(mgs,il))))*((4.0f+alpha(mgs,il)))),(((((3.0f+alpha(mgs,il)))*((2.0f+alpha(mgs,il))))*((1.0f+alpha(mgs,il))))));
// WRF:8428
if((rescale_high_alpha&&(alp>=(alphamax-0.01f)))) {
// WRF:8429
cx(mgs,il) = (lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(mgs,il))))),2));
// WRF:8430
an(igs(mgs),jy,kgs(mgs),coeff_ln{}(il)) = cx(mgs,il);
// WRF:8432
} else if(((rescale_low_alpha&&(alp<=alphamin))&&(!(((il==lh)&&(icvhl2h>0)))))) {
// WRF:8435
z1 = lake_div((((g1*lake_pow(dn(igs(mgs),jy,kgs(mgs)),2))*(qr))*qr),chw);
// WRF:8436
z = (z1*lake_pow((lake_div(6.f,((pi*xdn(mgs,il))))),2));
// WRF:8437
zx(mgs,il) = z;
// WRF:8438
an(igs(mgs),jy,kgs(mgs),coeff_lz{}(il)) = z;
// WRF:8439
}
// WRF:8440
}
// WRF:8441
} else {
// WRF:8442
}
// WRF:8443
}
// WRF:8444
}
// WRF:8446
}
// WRF:8448
}
// WRF:8452
}
// WRF:8454
if((lzhl>1)) {
// WRF:8455
if((lhl>1)) {
// WRF:8457
}
// WRF:8458
}
// WRF:8465
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8468
setvtz(ngscnt,qx,qxmin,qxw,cx,rho0,rhovt,xdia,cno,cnostmp,xmas,vtxbar,xdn,xvmn,xvmx,xv,cdx,cdxgs,ipconc,ndebugzf,ngs,nz,kgs,fadvisc,cwmasn,cwmasx,cwradn,cnina,cimn,cimx,itype1,itype2,temcg,infdo,alpha,ildo,axx,bxx);
// WRF:8480
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:8481
for(mgs=1;mgs<=ngscnt;mgs+=1) {
// WRF:8483
vtmax = 150.0f;
// WRF:8486
if(((vtxbar(mgs,il,2)>vtxbar(mgs,il,1))||(((vtxbar(mgs,il,1)>vtxbar(mgs,il,3))&&(vtxbar(mgs,il,3)>0.0f))))) {
// WRF:8504
vtxbar(mgs,il,1) = lake_max(vtxbar(mgs,il,1),vtxbar(mgs,il,2));
// WRF:8505
vtxbar(mgs,il,3) = lake_max(vtxbar(mgs,il,3),vtxbar(mgs,il,1));
// WRF:8507
}
// WRF:8510
if((((vtxbar(mgs,il,1)>vtmax)||(vtxbar(mgs,il,2)>vtmax))||(vtxbar(mgs,il,3)>vtmax))) {
// WRF:8525
vtxbar(mgs,il,1) = lake_min(vtmax,vtxbar(mgs,il,1));
// WRF:8526
vtxbar(mgs,il,2) = lake_min(vtmax,vtxbar(mgs,il,2));
// WRF:8527
vtxbar(mgs,il,3) = lake_min(vtmax,vtxbar(mgs,il,3));
// WRF:8530
}
// WRF:8533
xvt(kgs(mgs),igs(mgs),1,il) = vtxbar(mgs,il,1);
// WRF:8534
xvt(kgs(mgs),igs(mgs),2,il) = vtxbar(mgs,il,2);
// WRF:8535
if((infdo>=2)) {
// WRF:8536
xvt(kgs(mgs),igs(mgs),3,il) = vtxbar(mgs,il,3);
// WRF:8537
} else {
// WRF:8538
xvt(kgs(mgs),igs(mgs),3,il) = 0.0f;
// WRF:8539
}
// WRF:8543
}
// WRF:8544
}
// WRF:8547
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8551
L9998: ;
;
// WRF:8553
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8555
if((kz>(nz-1))) {
// WRF:8556
goto L1200;
// WRF:8557
} else {
// WRF:8558
nzmpb = kz;
// WRF:8559
}
// WRF:8561
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8563
}
// WRF:8565
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8567
L1200: ;
;
// WRF:8574
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8577
return;
}
__device__ __forceinline__ void ziegfall_point(int nx,int ny,int nz,int nor,int norz,int na,float dtp,int jgs,int ixcol,Array<float> xvt,Array<const float> rhovtzx,Array<float> an,Array<const float> dn,int ipconc0,Array<const float> t0,Array<const float> t7,float cwmasn,float cwmasx,float cwradn,Array<const float> qxmin,Array<const float> xdnmx,Array<const float> xdnmn,Array<const float> cdx,Array<const float> cno,Array<const float> xdn0,Array<const float> xvmn,Array<const float> xvmx,int ngs,Array<float> qx,Array<const float> qxw,Array<float> cx,Array<float> xv,Array<float> vtxbar,Array<float> xmas,Array<float> xdn,Array<float> xdia,Array<float> vx,Array<float> alpha,Array<float> zx,Array<int> igs,Array<int> kgs,Array<float> rho0,Array<float> temcg,Array<float> temg,Array<float> rhovt,Array<const float> cwnc,Array<const float> cinc,Array<float> fadvisc,Array<const float> cwdia,Array<const float> cipmas,Array<float> cnina,Array<const float> cimas,Array<float> cnostmp,int infdo,int ildo,double timesetvt=0) {
constexpr int ng1=1;
int ngt;
float dtz1;
constexpr int ndebugzf=0;
int ix;
int jy;
int kz;
int i;
int j;
int k;
int il;
int ngscnt;

FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cdxgs{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> axx{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> bxx{};
float cwccn0;
int nxmpb;
int nzmpb;
int nxz;
int numgs;
int inumgs;
constexpr int kstag=1;
bool flag;
bool ldoliq;
float chw;
float qr;
float z;
float rd;
float alp;
float z1;
float g1;
float vr;
float nrx;
float tmp;
float vtmax;
float xvbarmax;
const float c1r=19.0f;
const float c2r=0.6f;
const float c3r=1.8f;
const float c4r=17.0f;
const float c1h=5.5f;
const float c2h=0.7f;
const float c3h=4.5f;
const float c4h=8.5f;
const float c1hl=3.7f;
const float c2hl=0.3f;
const float c3hl=9.0f;
const float c4hl=6.5f;
const float c5hl=1.0f;
const float c6hl=6.5f;
int l1;
int l2;
double dpt1;
double dpt2;
int ixb;
int jyb;
int kzb;
int ixe;
int jye;
int kze;
bool debug_mpi=false;
// WRF:7738
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:7748
ldoliq = false;
// WRF:7749
if((ls>1)) {
// WRF:7750
#pragma unroll
for(il=ls;il<=lhab;il+=1) {
// WRF:7751
ldoliq = (ldoliq||((coeff_lliq{}(il)>1)));
// WRF:7752
}
// WRF:7753
}
// WRF:7787
jy = jgs;
// WRF:7788
nxmpb = ixcol;
// WRF:7789
nzmpb = 1;
// WRF:7790
nxz = (1*nz);
// WRF:7792
numgs = 1;
// WRF:7794
if((ildo==0)) {
// WRF:7795
l1 = lc;
// WRF:7796
l2 = lhab;
// WRF:7797
} else {
// WRF:7798
l1 = ildo;
// WRF:7799
l2 = ildo;
// WRF:7800
}
{ix=kz=1;flag=false;
#pragma unroll
for(il=l1;il<=l2;++il)flag=flag||(an(1,jy,1,il)>coeff_qxmin{}(il));
if(!flag)return;ngscnt=1;
// WRF:7841
for(int _once=0;_once<1;++_once) {
// WRF:7843
rho0(1) = dn(1,jy,1);
// WRF:7844
rhovt(1) = rhovtzx(1,ixcol);
// WRF:7845
temg(1) = t0(1,jy,1);
// WRF:7846
temcg(1) = (temg(1)-tfr);
// WRF:7850
}
// WRF:7853
if(((lc>1)&&(((ildo==0)||(ildo==lc))))) {
// WRF:7854
for(int _once=0;_once<1;++_once) {
// WRF:7855
fadvisc(1) = ((advisc0*(lake_div(416.16f,((temg(1)+120.0f)))))*lake_pow((lake_div(temg(1),296.0f)),(1.5f)));
// WRF:7857
}
// WRF:7858
}
// WRF:7860
if((ipconc==0)) {
// WRF:7861
for(int _once=0;_once<1;++_once) {
// WRF:7862
cnina(1) = t7(1,jgs,1);
// WRF:7863
}
// WRF:7864
}
// WRF:7867
if((ildo>0)) {
// WRF:7868
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s2=1;_s2<=3;++_s2) {vtxbar(_s0,ildo,_s2) = 0.0f;}}
// WRF:7869
} else {
// WRF:7870
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {for(int _s2=1;_s2<=3;++_s2) {vtxbar(_s0,_s1,_s2) = 0.0f;}}}
// WRF:7871
}
// WRF:7876
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:7877
for(int _once=0;_once<1;++_once) {
// WRF:7878
qx(1,il) = lake_max(an(1,jy,1,il),0.0f);
// WRF:7879
}
// WRF:7880
}
// WRF:7882
for(int _s0=1;_s0<=ngs;++_s0) {cnostmp(_s0) = coeff_cno{}(ls);}
// WRF:7883
if((((ipconc<1)&&lwsm6)&&(((ildo==0)||(ildo==ls))))) {
// WRF:7884
for(int _once=0;_once<1;++_once) {
// WRF:7885
tmp = lake_min(0.0f,temcg(1));
// WRF:7886
cnostmp(1) = lake_min(2.e8f,(2.e6f*gfk_exp((0.12f*tmp))));
// WRF:7887
}
// WRF:7888
}
// WRF:7894
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {cx(_s0,_s1) = 0.0f;}}
// WRF:7896
if((((ipconc>=1)&&(li>1))&&(((ildo==0)||(ildo==li))))) {
// WRF:7897
for(int _once=0;_once<1;++_once) {
// WRF:7898
cx(1,li) = lake_max(an(1,jy,1,lni),0.0f);
// WRF:7899
}
// WRF:7900
}
// WRF:7901
if((((ipconc>=2)&&(lc>1))&&(((ildo==0)||(ildo==lc))))) {
// WRF:7902
for(int _once=0;_once<1;++_once) {
// WRF:7903
cx(1,lc) = lake_max(an(1,jy,1,lnc),0.0f);
// WRF:7905
}
// WRF:7906
}
// WRF:7907
if((((ipconc>=3)&&(lr>1))&&(((ildo==0)||(ildo==lr))))) {
// WRF:7908
for(int _once=0;_once<1;++_once) {
// WRF:7909
cx(1,lr) = lake_max(an(1,jy,1,lnr),0.0f);
// WRF:7914
}
// WRF:7915
}
// WRF:7916
if((((ipconc>=4)&&(ls>1))&&(((ildo==0)||(ildo==ls))))) {
// WRF:7917
for(int _once=0;_once<1;++_once) {
// WRF:7918
cx(1,ls) = lake_max(an(1,jy,1,lns),0.0f);
// WRF:7923
}
// WRF:7924
}
// WRF:7926
if((((ipconc>=5)&&(lh>1))&&(((ildo==0)||(ildo==lh))))) {
// WRF:7927
for(int _once=0;_once<1;++_once) {
// WRF:7929
cx(1,lh) = lake_max(an(1,jy,1,lnh),0.0f);
// WRF:7935
}
// WRF:7936
}
// WRF:7938
if((((ipconc>=5)&&(lhl>1))&&(((ildo==0)||(ildo==lhl))))) {
// WRF:7939
for(int _once=0;_once<1;++_once) {
// WRF:7941
cx(1,lhl) = lake_max(an(1,jy,1,lnhl),0.0f);
// WRF:7950
}
// WRF:7951
}
// WRF:7953
for(int _once=0;_once<1;++_once) {
// WRF:7954
xdn(1,lc) = coeff_xdn0{}(lc);
// WRF:7955
xdn(1,lr) = coeff_xdn0{}(lr);
// WRF:7958
if((li>1)) {xdn(1,li) = coeff_xdn0{}(li);}
// WRF:7959
if((ls>1)) {xdn(1,ls) = coeff_xdn0{}(ls);}
// WRF:7960
if((lh>1)) {xdn(1,lh) = coeff_xdn0{}(lh);}
// WRF:7961
if((lhl>1)) {xdn(1,lhl) = coeff_xdn0{}(lhl);}
// WRF:7962
}
// WRF:7967
if((ldovol&&(((ildo==0)||(ildo>=li))))) {
// WRF:7969
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=li;_s1<=lhab;++_s1) {vx(_s0,_s1) = 0.0f;}}
// WRF:7971
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:7973
if((coeff_lvol{}(il)>=1)) {
// WRF:7975
for(int _once=0;_once<1;++_once) {
// WRF:7976
vx(1,il) = lake_max(an(1,jy,1,coeff_lvol{}(il)),0.0f);
// WRF:7977
if(((vx(1,il)>((rho0(1)*coeff_qxmin{}(il))*1.e-3f))&&(qx(1,il)>coeff_qxmin{}(il)))) {
// WRF:7978
xdn(1,il) = lake_min(coeff_xdnmx{}(il),lake_max(coeff_xdnmn{}(il),lake_div((rho0(1)*qx(1,il)),vx(1,il))));
// WRF:7979
}
// WRF:7980
}
// WRF:7982
}
// WRF:7984
}
// WRF:7986
}
// WRF:7988
#pragma unroll
for(il=lg;il<=lhab;il+=1) {
// WRF:7989
for(int _once=0;_once<1;++_once) {
// WRF:7990
alpha(1,il) = coeff_dnu{}(il);
// WRF:7991
}
// WRF:7992
}
// WRF:7994
if((imurain==1)) {
// WRF:7995
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = alphar;}
// WRF:7996
} else if((imurain==3)) {
// WRF:7997
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = coeff_xnu{}(lr);}
// WRF:7998
}
// WRF:8001
if(((ipconc==5)&&(imydiagalpha>0))) {
// WRF:8002
for(int _once=0;_once<1;++_once) {
// WRF:8003
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(cx(1,lr)>cxmin))) {
// WRF:8004
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*cx(1,lr))));
// WRF:8005
xdia(1,lr,3) = lake_pow((((xv(1,lr)*6.0f)*cwc1)),(lake_div(1.f,3.f)));
// WRF:8006
alpha(1,lr) = lake_min(alphamax,((c1r*tanhf((c2r*(((xdia(1,lr,3)*1000.f)-c3r)))))+c4r));
// WRF:8007
}
// WRF:8008
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(cx(1,lh)>cxmin))) {
// WRF:8009
xv(1,lh) = lake_div((rho0(1)*qx(1,lh)),((xdn(1,lh)*cx(1,lh))));
// WRF:8010
xdia(1,lh,3) = lake_pow((((xv(1,lh)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:8011
alpha(1,lh) = lake_min(alphamax,((c1h*tanhf((c2h*(((xdia(1,lh,3)*1000.f)-c3h)))))+c4h));
// WRF:8012
}
// WRF:8015
if((lhl>0)) {
// WRF:8016
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(cx(1,lhl)>cxmin))) {
// WRF:8017
xv(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),((xdn(1,lhl)*cx(1,lhl))));
// WRF:8018
xdia(1,lhl,3) = lake_pow((((xv(1,lhl)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:8019
if((xdia(1,lhl,3)<0.008f)) {
// WRF:8020
alpha(1,lhl) = lake_min(alphamax,((c1hl*tanhf((c2hl*(((xdia(1,lhl,3)*1000.f)-c3hl)))))+c4hl));
// WRF:8021
} else {
// WRF:8022
alpha(1,lhl) = lake_min(alphamax,(((c5hl*xdia(1,lhl,3))*1000.f)+c6hl));
// WRF:8023
}
// WRF:8024
}
// WRF:8025
}
// WRF:8026
}
// WRF:8027
}
// WRF:8033
if(((ipconc>=6)||(lzr>1))) {
// WRF:8035
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lr;_s1<=lhab;++_s1) {zx(_s0,_s1) = 0.0f;}}
// WRF:8038
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:8040
if((coeff_lz{}(il)>=1)) {
// WRF:8042
for(int _once=0;_once<1;++_once) {
// WRF:8043
zx(1,il) = lake_max(an(1,jy,1,coeff_lz{}(il)),0.0f);
// WRF:8044
}
// WRF:8047
}
// WRF:8049
}
// WRF:8051
}
// WRF:8060
if((((coeff_lz{}(lr)>1)&&(((ildo==0)||(ildo==lr))))&&(imurain==3))) {
// WRF:8061
il = lr;
// WRF:8062
for(int _once=0;_once<1;++_once) {
// WRF:8064
if(((iresetmoments==1)||(iresetmoments==il))) {
// WRF:8066
if((zx(1,lr)<=zxmin)) {
// WRF:8067
qx(1,lr) = 0.0f;
// WRF:8068
cx(1,lr) = 0.0f;
// WRF:8069
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,lr));
// WRF:8070
an(1,jgs,1,lr) = qx(1,lr);
// WRF:8071
an(1,jgs,1,coeff_ln{}(lr)) = cx(1,lr);
// WRF:8074
} else if((cx(1,lr)<=cxmin)) {
// WRF:8075
zx(1,lr) = 0.0f;
// WRF:8076
qx(1,lr) = 0.0f;
// WRF:8077
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,lr));
// WRF:8078
an(1,jgs,1,lr) = qx(1,lr);
// WRF:8079
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:8080
}
// WRF:8081
}
// WRF:8085
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:8087
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*lake_max(1.0e-11f,cx(1,lr)))));
// WRF:8088
if((xv(1,lr)>coeff_xvmx{}(lr))) {
// WRF:8098
} else if((xv(1,lr)<coeff_xvmn{}(lr))) {
// WRF:8099
xv(1,lr) = coeff_xvmn{}(lr);
// WRF:8100
cx(1,lr) = lake_div((rho0(1)*qx(1,lr)),((coeff_xvmn{}(lr)*xdn(1,lr))));
// WRF:8101
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8102
}
// WRF:8104
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:8106
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8107
z = zx(1,il);
// WRF:8108
qr = qx(1,il);
// WRF:8110
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:8111
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8113
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>cxmin))) {
// WRF:8115
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8116
chw = cx(1,il);
// WRF:8117
qr = qx(1,il);
// WRF:8126
zx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),((lake_pow(xdn(1,lr),2)*chw)));
// WRF:8127
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8129
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:8133
if((qx(1,il)<1.e-8f)) {
// WRF:8134
qx(1,il) = 0.0f;
// WRF:8135
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8136
an(1,jgs,1,il) = qx(1,il);
// WRF:8137
} else {
// WRF:8139
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:8140
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8142
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8143
z = zx(1,il);
// WRF:8144
qr = qx(1,il);
// WRF:8145
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:8146
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8147
}
// WRF:8148
}
// WRF:8150
if((zx(1,lr)>0.0f)) {
// WRF:8151
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((1000.f*lake_max(1.0e-9f,cx(1,lr)))));
// WRF:8152
vr = xv(1,lr);
// WRF:8154
qr = qx(1,lr);
// WRF:8155
nrx = cx(1,lr);
// WRF:8156
z = zx(1,lr);
// WRF:8162
if((z>0.0f)) {
// WRF:8164
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8165
for(i=1;i<=20;i+=1) {
// WRF:8167
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:8168
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8169
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8171
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8172
}
// WRF:8177
if((xv(1,il)>coeff_xvmx{}(il))) {
// WRF:8178
tmp = cx(1,il);
// WRF:8179
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:8180
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:8181
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:8182
if((tmp<cx(1,il))) {
// WRF:8184
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8185
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:8186
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8188
vr = xv(1,lr);
// WRF:8189
qr = qx(1,lr);
// WRF:8190
nrx = cx(1,lr);
// WRF:8191
z = zx(1,lr);
// WRF:8195
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8196
for(i=1;i<=20;i+=1) {
// WRF:8197
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:8198
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8199
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:8200
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:8201
}
// WRF:8204
}
// WRF:8205
}
// WRF:8212
if((true&&((((alpha(1,il)<=rnumin)||(alp==rnumin))||(alp==rnumax))))) {
// WRF:8214
if((rescale_high_alpha&&(alp>=(rnumax-0.01f)))) {
// WRF:8215
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8216
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(1.f,(xdn(1,il)))),2));
// WRF:8217
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8219
} else if((rescale_low_alphar&&(alp<=rnumin))) {
// WRF:8221
z = lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:8222
zx(1,il) = z;
// WRF:8223
an(1,jy,1,coeff_lz{}(il)) = z;
// WRF:8225
}
// WRF:8226
}
// WRF:8228
}
// WRF:8229
}
// WRF:8231
} else {
// WRF:8233
zx(1,lr) = 0.0f;
// WRF:8234
cx(1,lr) = 0.0f;
// WRF:8235
an(1,jgs,1,coeff_ln{}(lr)) = cx(1,lr);
// WRF:8236
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:8238
}
// WRF:8240
}
// WRF:8241
}
// WRF:8244
if((ipconc>=6)) {
// WRF:8248
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:8250
if((((coeff_lz{}(il)>1)&&(((ildo==0)||(ildo==il))))&&((!(((il==lr)&&(imurain==3))))))) {
// WRF:8252
for(int _once=0;_once<1;++_once) {
// WRF:8254
if((((iresetmoments==1)||(iresetmoments==il))||(iresetmoments==(-1)))) {
// WRF:8255
if((zx(1,il)<=zxmin)) {
// WRF:8256
qx(1,il) = 0.0f;
// WRF:8257
cx(1,il) = 0.0f;
// WRF:8258
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8259
an(1,jgs,1,il) = qx(1,il);
// WRF:8260
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8261
} else if(((iresetmoments==(-1))&&(qx(1,il)<coeff_qxmin{}(il)))) {
// WRF:8262
zx(1,il) = 0.0f;
// WRF:8263
cx(1,il) = 0.0f;
// WRF:8264
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8266
qx(1,il) = 0.0f;
// WRF:8267
an(1,jgs,1,il) = qx(1,il);
// WRF:8268
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8269
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8271
} else if(((cx(1,il)<=cxmin)&&(iresetmoments!=(-1)))) {
// WRF:8273
zx(1,il) = 0.0f;
// WRF:8274
qx(1,il) = 0.0f;
// WRF:8275
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8276
an(1,jgs,1,il) = qx(1,il);
// WRF:8277
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8278
}
// WRF:8279
}
// WRF:8281
if(((zx(1,il)<=zxmin)&&(cx(1,il)<=cxmin))) {
// WRF:8282
zx(1,il) = 0.0f;
// WRF:8283
cx(1,il) = 0.0f;
// WRF:8284
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8285
qx(1,il) = 0.0f;
// WRF:8286
an(1,jgs,1,il) = qx(1,il);
// WRF:8287
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8288
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8289
}
// WRF:8291
if((qx(1,il)>coeff_qxmin{}(il))) {
// WRF:8293
xv(1,il) = lake_div((rho0(1)*qx(1,il)),((xdn(1,il)*lake_max(1.0e-9f,cx(1,il)))));
// WRF:8294
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:8296
if((xv(1,il)<coeff_xvmn{}(il))) {
// WRF:8298
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:8299
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:8300
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:8308
}
// WRF:8310
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:8312
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:8314
z = zx(1,il);
// WRF:8315
qr = qx(1,il);
// WRF:8316
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:8317
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8319
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>cxmin))) {
// WRF:8321
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:8323
chw = cx(1,il);
// WRF:8324
qr = qx(1,il);
// WRF:8326
zx(1,il) = lake_min((zxmin*1.1f),lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6*qr)),2)),((chw*lake_pow(((pi*xdn(1,il))),2)))));
// WRF:8327
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8328
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:8334
if((qx(1,il)<1.e-8f)) {
// WRF:8335
qx(1,il) = 0.0f;
// WRF:8336
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:8337
an(1,jgs,1,il) = qx(1,il);
// WRF:8338
} else {
// WRF:8342
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:8343
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8345
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:8347
z = zx(1,il);
// WRF:8348
qr = qx(1,il);
// WRF:8349
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:8350
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8351
}
// WRF:8352
}
// WRF:8353
}
// WRF:8355
if(((qx(1,il)>coeff_qxmin{}(il))&&(cx(1,il)>cxmin))) {
// WRF:8356
chw = cx(1,il);
// WRF:8357
qr = qx(1,il);
// WRF:8358
z = zx(1,il);
// WRF:8360
if((zx(1,il)>0.f)) {
// WRF:8363
rd = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((dn(1,jy,1)*qr)),2)));
// WRF:8365
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rd)))-1.0f);
// WRF:8367
for(i=1;i<=10;i+=1) {
// WRF:8368
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:8369
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8370
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rd)))-1.0f);
// WRF:8373
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8374
}
// WRF:8380
if((imaxdiaopt==1)) {
// WRF:8381
xvbarmax = coeff_xvmx{}(il);
// WRF:8382
} else if((imaxdiaopt==2)) {
// WRF:8383
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((3.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:8384
} else if((imaxdiaopt==3)) {
// WRF:8385
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((4.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:8386
}
// WRF:8388
if((xv(1,il)>xvbarmax)) {
// WRF:8389
tmp = cx(1,il);
// WRF:8390
xv(1,il) = lake_min(xvbarmax,lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:8391
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:8392
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:8393
if((tmp<cx(1,il))) {
// WRF:8394
g1 = lake_div((((36.f*((6.0f+alpha(1,il))))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),((((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))*lake_pow(pi,2))));
// WRF:8396
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:8397
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:8399
chw = cx(1,il);
// WRF:8400
qr = qx(1,il);
// WRF:8401
z = zx(1,il);
// WRF:8403
rd = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((rho0(1)*qr)),2)));
// WRF:8404
alp = (lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rd)))-1.0f);
// WRF:8406
for(i=1;i<=10;i+=1) {
// WRF:8407
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:8408
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8409
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rd)))-1.0f);
// WRF:8411
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:8412
}
// WRF:8415
}
// WRF:8416
}
// WRF:8422
if((((rescale_low_alpha||rescale_high_alpha))&&((((alpha(1,il)<=alphamin)||(alp==alphamin))||(alp==alphamax))))) {
// WRF:8425
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:8428
if((rescale_high_alpha&&(alp>=(alphamax-0.01f)))) {
// WRF:8429
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:8430
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:8432
} else if(((rescale_low_alpha&&(alp<=alphamin))&&(!(((il==lh)&&(icvhl2h>0)))))) {
// WRF:8435
z1 = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),chw);
// WRF:8436
z = (z1*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:8437
zx(1,il) = z;
// WRF:8438
an(1,jy,1,coeff_lz{}(il)) = z;
// WRF:8439
}
// WRF:8440
}
// WRF:8441
} else {
// WRF:8442
}
// WRF:8443
}
// WRF:8444
}
// WRF:8446
}
// WRF:8448
}
// WRF:8452
}
// WRF:8454
if((lzhl>1)) {
// WRF:8455
if((lhl>1)) {
// WRF:8457
}
// WRF:8458
}
// WRF:8465
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8468
setvtz_point(ngscnt,qx,qxmin,qxw,cx,rho0,rhovt,xdia,cno,cnostmp,xmas,vtxbar,xdn,xvmn,xvmx,xv,cdx,cdxgs,ipconc,ndebugzf,ngs,nz,kgs,fadvisc,cwmasn,cwmasx,cwradn,cnina,cimn,cimx,itype1,itype2,temcg,infdo,alpha,ildo,axx,bxx);
// WRF:8480
#pragma unroll
for(il=l1;il<=l2;il+=1) {
// WRF:8481
for(int _once=0;_once<1;++_once) {
// WRF:8483
vtmax = 150.0f;
// WRF:8486
if(((vtxbar(1,il,2)>vtxbar(1,il,1))||(((vtxbar(1,il,1)>vtxbar(1,il,3))&&(vtxbar(1,il,3)>0.0f))))) {
// WRF:8504
vtxbar(1,il,1) = lake_max(vtxbar(1,il,1),vtxbar(1,il,2));
// WRF:8505
vtxbar(1,il,3) = lake_max(vtxbar(1,il,3),vtxbar(1,il,1));
// WRF:8507
}
// WRF:8510
if((((vtxbar(1,il,1)>vtmax)||(vtxbar(1,il,2)>vtmax))||(vtxbar(1,il,3)>vtmax))) {
// WRF:8525
vtxbar(1,il,1) = lake_min(vtmax,vtxbar(1,il,1));
// WRF:8526
vtxbar(1,il,2) = lake_min(vtmax,vtxbar(1,il,2));
// WRF:8527
vtxbar(1,il,3) = lake_min(vtmax,vtxbar(1,il,3));
// WRF:8530
}
// WRF:8533
xvt(1,1,1,il) = vtxbar(1,il,1);
// WRF:8534
xvt(1,1,2,il) = vtxbar(1,il,2);
// WRF:8535
if((infdo>=2)) {
// WRF:8536
xvt(1,1,3,il) = vtxbar(1,il,3);
// WRF:8537
} else {
// WRF:8538
xvt(1,1,3,il) = 0.0f;
// WRF:8539
}
// WRF:8543
}
// WRF:8544
}
// WRF:8547
if((ndebugzf>0)) {; /* native diagnostic only */}
// WRF:8551
L9998: ;
;
// WRF:8553
if((ndebugzf>0)) {; /* native diagnostic only */}
return;
}
}
__device__ __forceinline__ void fallout1d(int nx,int ny,int nz,int nor,int na,float dtp,float dtfrac,int jgs,Array<const float> vt,Array<float> a,Array<const float> db1,int ia,int id,Array<float> xfall,Array<const float> dtz1,int ixcol) {
int ngt;
constexpr int ng1=1;
float cmax;
int ix;
int jy;
int kz;
int n;
int k;
int iv1;
int iv2;
float tmp;
int imn;
int imx;
int kmn;
int kmx;
FixedStorage<float,((NZ+1)-1+1),1,(NZ+1)> qtmp1{};
int ixb;
int jyb;
int kzb;
int ixe;
int jye;
int kze;
bool debug_mpi=true;
// WRF:4791
jy = 1;
// WRF:4793
iv1 = 0;
// WRF:4794
iv2 = 0;
// WRF:4796
imn = nx;
// WRF:4797
imx = 1;
// WRF:4798
kmn = nz;
// WRF:4799
kmx = 1;
// WRF:4801
cmax = 0.0f;
// WRF:4803
kzb = 1;
// WRF:4804
kze = nz;
// WRF:4806
ixb = ixcol;
// WRF:4807
ixe = ixcol;
// WRF:4808
ix = ixcol;
// WRF:4810
qtmp1((nz+1)) = 0.0f;
// WRF:4812
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4816
if((id==1)) {
// WRF:4817
qtmp1(kz) = ((a(ix,jgs,kz,ia)*vt(kz,ix))*db1(ix,kz));
// WRF:4818
} else {
// WRF:4819
qtmp1(kz) = (a(ix,jgs,kz,ia)*vt(kz,ix));
// WRF:4820
}
// WRF:4822
if((a(ix,jgs,kz,ia)!=0.0f)) {
// WRF:4825
kmn = lake_min(kz,kmn);
// WRF:4826
kmx = lake_max(kz,kmx);
// WRF:4827
}
// WRF:4829
}
// WRF:4831
kmn = lake_max(1,(kmn-1));
// WRF:4838
if((kmn==1)) {
// WRF:4840
kz = 1;
// WRF:4842
xfall(ix,jy,ia) = (xfall(ix,jy,ia)+((a(ix,jgs,kz,ia)*vt(kz,ix))*dtfrac));
// WRF:4845
}
// WRF:4847
for(kz=1;kz<=nz;kz+=1) {
// WRF:4849
a(ix,jgs,kz,ia) = (a(ix,jgs,kz,ia)+((dtp*dtz1(kz,ix,id))*((qtmp1((kz+1))-qtmp1(kz)))));
// WRF:4851
}
// WRF:4854
return;
}
__device__ __forceinline__ void calczgr1d(int nx,int ny,int nz,int nor,int na,Array<const float> a,int ixe,int kze,Array<float> z,Array<const float> db,int jgs,int ipconc,float alpha,int l,int ln,float qmin,float xvmn,float xvmx,int lvol,float rho_qx,int ixcol) {
int ngt;
constexpr int norz=3;
int ix;
int jy;
int kz;
float vr;
float qr;
float nrx;
float rd;
float xv;
float g1;
float zx;
float chw;
float xdn;
float ynu;
// WRF:4889
jy = jgs;
// WRF:4890
ix = ixcol;
// WRF:4892
if(((((l==lh)||(l==lhl))||(((l==lr)&&(imurain==1))))||(((l==ls)&&(imusnow==1))))) {
// WRF:4896
for(kz=1;kz<=kze;kz+=1) {
// WRF:4900
if(((a(ix,jy,kz,l)>qmin)&&(a(ix,jy,kz,ln)>1.e-15f))) {
// WRF:4902
if((lvol>1)) {
// WRF:4903
if((a(ix,jy,kz,lvol)>0.0f)) {
// WRF:4904
xdn = lake_div((db(ix,kz)*a(ix,jy,kz,l)),a(ix,jy,kz,lvol));
// WRF:4905
xdn = lake_min(900.f,lake_max(hdnmn,xdn));
// WRF:4906
} else {
// WRF:4907
xdn = rho_qx;
// WRF:4908
}
// WRF:4909
} else {
// WRF:4910
xdn = rho_qx;
// WRF:4911
}
// WRF:4913
if((l==lr)) {xdn = 1000.f;}
// WRF:4915
qr = a(ix,jy,kz,l);
// WRF:4916
xv = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xdn*a(ix,jy,kz,ln))));
// WRF:4917
chw = a(ix,jy,kz,ln);
// WRF:4919
if(((xv<xvmn)||(xv>xvmx))) {
// WRF:4920
xv = lake_min(xvmx,lake_max(xvmn,xv));
// WRF:4921
chw = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xv*xdn)));
// WRF:4922
}
// WRF:4924
g1 = lake_div(((((6.0f+alpha))*((5.0f+alpha)))*((4.0f+alpha))),(((((3.0f+alpha))*((2.0f+alpha)))*((1.0f+alpha)))));
// WRF:4926
zx = lake_div((((g1*lake_pow(db(ix,kz),2))*(a(ix,jy,kz,l)))*a(ix,jy,kz,l)),chw);
// WRF:4928
z(ix,kz,l) = (zx*lake_pow((lake_div(6.f,((pi*1000.f)))),2));
// WRF:4935
} else {
// WRF:4937
z(ix,kz,l) = 0.0f;
// WRF:4939
}
// WRF:4941
}
// WRF:4943
} else if(((((l==ls)&&(imusnow==3)))||(((l==lr)&&(imurain==3))))) {
// WRF:4945
xdn = rho_qx;
// WRF:4946
if((l==ls)) {ynu = snu;}
// WRF:4947
if((l==lr)) {ynu = rnu;}
// WRF:4949
for(kz=1;kz<=kze;kz+=1) {
// WRF:4951
if(((a(ix,jy,kz,l)>qmin)&&(a(ix,jy,kz,ln)>1.e-15f))) {
// WRF:4953
vr = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xdn*a(ix,jy,kz,ln))));
// WRF:4955
z(ix,kz,l) = lake_div((((3.6f*((ynu+2.0f)))*a(ix,jy,kz,ln))*lake_pow(vr,2)),((ynu+1.0f)));
// WRF:4959
} else {
// WRF:4961
z(ix,kz,l) = 0.0f;
// WRF:4963
}
// WRF:4966
}
// WRF:4968
}
// WRF:4970
return;
}
__device__ __forceinline__ void calcnfromz1d(int nx,int ny,int nz,int nor,int na,Array<float> a,Array<const float> t0,int ixe,int kze,Array<const float> z0,Array<const float> db,int jgs,int ipconc,float alpha,int l,int ln,float qmin,float xvmn,float xvmx,Array<const float> t1,int lvol,float rho_qx,int infall,int ixcol) {
int ngt;
int ix;
int jy;
int kz;
double vr;
double qr;
double nrx;
double rd;
double g1;
double zx;
double chw;
double z;
double znew;
double zt;
double zxt;
float xv;
float xdn;
int ndbz;
int nmwgt;
int nnwgt;
int nwlessthanz;
// WRF:5016
ndbz = 0;
// WRF:5017
nmwgt = 0;
// WRF:5018
nnwgt = 0;
// WRF:5019
nwlessthanz = 0;
// WRF:5023
jy = jgs;
// WRF:5024
ix = ixcol;
// WRF:5026
if((((l==lh)||(l==lhl))||(((l==lr)&&(imurain==1))))) {
// WRF:5028
g1 = lake_div(((((6.0f+alpha))*((5.0f+alpha)))*((4.0f+alpha))),(((((3.0f+alpha))*((2.0f+alpha)))*((1.0f+alpha)))));
// WRF:5031
for(kz=1;kz<=kze;kz+=1) {
// WRF:5034
if((t0(ix,jy,kz)>0.f)) {
// WRF:5036
if((lvol>1)) {
// WRF:5037
if((a(ix,jy,kz,lvol)>0.0f)) {
// WRF:5038
xdn = lake_div((db(ix,kz)*a(ix,jy,kz,l)),a(ix,jy,kz,lvol));
// WRF:5039
xdn = lake_min(900.f,lake_max(hdnmn,xdn));
// WRF:5040
} else {
// WRF:5041
xdn = rho_qx;
// WRF:5042
}
// WRF:5043
} else {
// WRF:5044
xdn = rho_qx;
// WRF:5045
}
// WRF:5047
if((l==lr)) {xdn = 1000.f;}
// WRF:5049
qr = a(ix,jy,kz,l);
// WRF:5050
xv = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xdn*a(ix,jy,kz,ln))));
// WRF:5051
chw = a(ix,jy,kz,ln);
// WRF:5053
if(((xv<xvmn)||(xv>xvmx))) {
// WRF:5054
xv = lake_min(xvmx,lake_max(xvmn,xv));
// WRF:5055
chw = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xv*xdn)));
// WRF:5056
}
// WRF:5058
zx = lake_div((((g1*lake_pow(db(ix,kz),2))*(a(ix,jy,kz,l)))*a(ix,jy,kz,l)),chw);
// WRF:5059
z = (zx*lake_pow((lake_div(6.f,((pi*1000.f)))),2));
// WRF:5062
if(((((z>t0(ix,jy,kz))&&(z>0.0f))&&(t0(ix,jy,kz)>z0(ix,kz,l))))) {
// WRF:5065
zx = lake_div(t0(ix,jy,kz),(lake_pow((lake_div(6.f,((pi*1000.f)))),2)));
// WRF:5067
nrx = lake_div((((g1*lake_pow(db(ix,kz),2))*(a(ix,jy,kz,l)))*a(ix,jy,kz,l)),zx);
// WRF:5068
if((infall==3)) {
// WRF:5069
if((nrx>a(ix,jy,kz,ln))) {
// WRF:5070
ndbz = (ndbz+1);
// WRF:5071
if((t1(ix,jy,kz)<ndbz)) {nwlessthanz = (nwlessthanz+1);}
// WRF:5072
} else {
// WRF:5073
nnwgt = (nnwgt+1);
// WRF:5074
}
// WRF:5075
a(ix,jy,kz,ln) = lake_max(((float)(nrx)),a(ix,jy,kz,ln));
// WRF:5076
} else {
// WRF:5077
if(((nrx>a(ix,jy,kz,ln))&&(t1(ix,jy,kz)>a(ix,jy,kz,ln)))) {
// WRF:5078
if((nrx<t1(ix,jy,kz))) {
// WRF:5079
ndbz = (ndbz+1);
// WRF:5080
} else {
// WRF:5081
nmwgt = (nmwgt+1);
// WRF:5082
if((t1(ix,jy,kz)<ndbz)) {nwlessthanz = (nwlessthanz+1);}
// WRF:5083
}
// WRF:5084
} else {
// WRF:5085
nnwgt = (nnwgt+1);
// WRF:5086
}
// WRF:5088
a(ix,jy,kz,ln) = lake_max(lake_min(((float)(nrx)),t1(ix,jy,kz)),a(ix,jy,kz,ln));
// WRF:5089
}
// WRF:5091
} else {
// WRF:5092
if(((t1(ix,jy,kz)>0)&&(a(ix,jy,kz,ln)>0))) {
// WRF:5093
if((t1(ix,jy,kz)>a(ix,jy,kz,ln))) {
// WRF:5094
nmwgt = (nmwgt+1);
// WRF:5095
} else {
// WRF:5096
nnwgt = (nnwgt+1);
// WRF:5097
}
// WRF:5098
}
// WRF:5099
a(ix,jy,kz,ln) = lake_max(t1(ix,jy,kz),a(ix,jy,kz,ln));
// WRF:5100
nrx = a(ix,jy,kz,ln);
// WRF:5104
}
// WRF:5107
} else {
// WRF:5108
if(((t1(ix,jy,kz)>0)&&(a(ix,jy,kz,ln)>0))) {
// WRF:5109
if((t1(ix,jy,kz)>a(ix,jy,kz,ln))) {
// WRF:5110
nmwgt = (nmwgt+1);
// WRF:5111
} else {
// WRF:5112
nnwgt = (nnwgt+1);
// WRF:5113
}
// WRF:5114
}
// WRF:5115
}
// WRF:5117
}
// WRF:5120
} else if(((l==lr)&&(imurain==3))) {
// WRF:5122
xdn = 1000.f;
// WRF:5124
for(kz=1;kz<=kze;kz+=1) {
// WRF:5125
if((t0(ix,jy,kz)>0.f)) {
// WRF:5127
vr = lake_div((db(ix,kz)*a(ix,jy,kz,l)),((xdn*a(ix,jy,kz,ln))));
// WRF:5128
z = lake_div((((3.6f*((rnu+2.0f)))*a(ix,jy,kz,ln))*lake_pow(vr,2)),((rnu+1.0f)));
// WRF:5130
if(((((z>t0(ix,jy,kz))&&(z>0.0f))&&(t0(ix,jy,kz)>0.0f))&&(t0(ix,jy,kz)>z0(ix,kz,l)))) {
// WRF:5134
vr = lake_div((db(ix,kz)*a(ix,jy,kz,l)),(xdn));
// WRF:5135
chw = a(ix,jy,kz,ln);
// WRF:5136
nrx = lake_div(((3.6f*((rnu+2.0f)))*lake_pow(vr,2)),((((rnu+1.0f))*t0(ix,jy,kz))));
// WRF:5137
if((infall==3)) {
// WRF:5138
a(ix,jy,kz,ln) = lake_max(((float)(nrx)),a(ix,jy,kz,ln));
// WRF:5139
} else if((infall==4)) {
// WRF:5140
a(ix,jy,kz,ln) = lake_max(lake_min(((float)(nrx)),t1(ix,jy,kz)),a(ix,jy,kz,ln));
// WRF:5141
}
// WRF:5143
} else {
// WRF:5145
a(ix,jy,kz,ln) = lake_max(t1(ix,jy,kz),a(ix,jy,kz,ln));
// WRF:5147
}
// WRF:5149
} else {
// WRF:5151
a(ix,jy,kz,ln) = lake_max(t1(ix,jy,kz),a(ix,jy,kz,ln));
// WRF:5153
}
// WRF:5156
}
// WRF:5158
}
// WRF:5160
return;
}
__device__ __forceinline__ void velocity1d(float dtp,int nx,int ny,int nz,Array<float> an,int na,int nor,int norz,Array<float> xfall,Array<const float> dn,Array<const float> dz3d,Array<const float> dz3dinv,Array<const float> t0,Array<const float> t7,int infdo,int jslab,int its,int jts,double timesed1,double timesed2,double timesed3,double zmaxsed,double timesetvt,Array<float> xvt) {
int ngt;
int jgs;
int ia;
int id;
constexpr int ng1=1;
int ix;
int jy;
int kz;
int ndfall;
int n;
int k;
int il;
int in;
float tmp;
float vtmax;
float dtptmp;
float dtfrac;
const float dz=200.f;
FixedStorage<float,(1-1+1)*((NZ+1)-1+1),1,1,1,(NZ+1)> db1{};
FixedStorage<float,((NZ+1)-1+1)*(1-1+1)*(1-0+1),1,(NZ+1),1,1,0,1> dtz1{};
FixedStorage<float,((NZ+1)-1+1)*(1-1+1),1,(NZ+1),1,1> dz2dinv{};
FixedStorage<float,(1-1+1)*((NZ+1)-1+1),1,1,1,(NZ+1)> db1inv{};
FixedStorage<float,(NZ-1+1)*(1-1+1),1,NZ,1,1> rhovtzx{};
FixedStorage<float,(1-1+1)*(1-1+1),1,1,1,1> xfall0{};

FixedStorage<float,((1+0)-((-0)+ng1)+1)*((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(1+0),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0)> tmpn{};
FixedStorage<float,((1+0)-((-0)+ng1)+1)*((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(1+0),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0)> tmpn2{};
FixedStorage<float,((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1)*(lhab-lr+1),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0),lr,lhab> z{};
double dummy;
double dt1;
double dt2;
double dt3;
double dt4;
int ngs;
int ngscnt;
int mgs;
int ipconc0;
FixedStorage<float,(1-1+1)*(lhab-lv+1),1,1,lv,lhab> qx{};
FixedStorage<float,(1-1+1)*(lhab-ls+1),1,1,ls,lhab> qxw{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xv{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(3-1+1),1,1,lc,lhab,1,3> vtxbar{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xmas{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xdn{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(3-1+1),1,1,lc,lhab,1,3> xdia{};
FixedStorage<float,(1-1+1)*(lhab-li+1),1,1,li,lhab> vx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> alpha{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> zx{};
FixedStorage<bool,(1-1+1)*(lhab-(lc+1)+1),1,1,(lc+1),lhab> hasmass{};
FixedStorage<int,(1-1+1),1,1> igs{};
FixedStorage<int,(1-1+1),1,1> kgs{};
FixedStorage<float,(1-1+1),1,1> rho0{};
FixedStorage<float,(1-1+1),1,1> temcg{};
FixedStorage<float,(1-1+1),1,1> temg{};
FixedStorage<float,(1-1+1),1,1> rhovt{};
FixedStorage<float,(1-1+1),1,1> cwnc{};
FixedStorage<float,(1-1+1),1,1> cinc{};
FixedStorage<float,(1-1+1),1,1> fadvisc{};
FixedStorage<float,(1-1+1),1,1> cwdia{};
FixedStorage<float,(1-1+1),1,1> cipmas{};
FixedStorage<float,(1-1+1),1,1> cnina{};
FixedStorage<float,(1-1+1),1,1> cimas{};
FixedStorage<float,(1-1+1),1,1> cnostmp{};
float cimasn;
float cimasx;
int ixb;
int jyb;
int kzb;
int ixe;
int jye;
int kze;
int plo;
int phi;
bool debug_mpi=true;
// WRF:4367
; /* automatic column storage */
// WRF:4368
; /* automatic column storage */
// WRF:4369
; /* automatic column storage */
// WRF:4371
ngs = 1;
// WRF:4373
; /* automatic column storage */
// WRF:4392
kzb = 1;
// WRF:4393
kze = nz;
// WRF:4395
ixb = 1;
// WRF:4396
ixe = nx;
// WRF:4399
jy = 1;
// WRF:4400
jgs = jy;
// WRF:4407
for(int _s0=1;_s0<=(nz+1);++_s0) {for(int _s1=1;_s1<=nx;++_s1) {for(int _s2=1;_s2<=3;++_s2) {for(int _s3=lc;_s3<=lhab;++_s3) {xvt(_s0,_s1,_s2,_s3) = 0.0f;}}}}
// WRF:4409
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4412
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4413
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4414
db1(ix,kz) = dn(ix,jy,kz);
// WRF:4415
db1inv(ix,kz) = lake_div(1.f,dn(ix,jy,kz));
// WRF:4416
rhovtzx(kz,ix) = sqrtf((rho00*lake_min(lake_div(1.0f,0.05f),db1inv(ix,kz))));
// WRF:4417
}
// WRF:4418
}
// WRF:4420
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4421
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4422
dtz1(kz,ix,0) = dz3dinv(ix,jy,kz);
// WRF:4423
dtz1(kz,ix,1) = (dz3dinv(ix,jy,kz)*db1inv(ix,kz));
// WRF:4424
dz2dinv(kz,ix) = dz3dinv(ix,jy,kz);
// WRF:4425
}
// WRF:4426
}
// WRF:4428
if((lzh>1)) {
// WRF:4429
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4430
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4431
an(ix,jy,kz,lzh) = lake_max(0.f,an(ix,jy,kz,lzh));
// WRF:4432
}
// WRF:4433
}
// WRF:4434
}
// WRF:4437
#pragma unroll
for(il=(lc+1);il<=lhab;il+=1) {
// WRF:4438
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4440
}
// WRF:4441
}
// WRF:4446
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4449
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4451
dummy = 0.e0;
// WRF:4454
ziegfall_point(nx,ny,nz,nor,norz,na,dtp,jgs,ix,xvt,rhovtzx,an,dn,ipconc,t0,t7,cwmasn,cwmasx,cwradn,qxmin,xdnmx,xdnmn,cdx,cno,xdn0,xvmn,xvmx,ngs,qx,qxw,cx,xv,vtxbar,xmas,xdn,xdia,vx,alpha,zx,igs,kgs,rho0,temcg,temg,rhovt,cwnc,cinc,fadvisc,cwdia,cipmas,cnina,cimas,cnostmp,infdo,0);
}
return;
}
__device__ __forceinline__ void sediment1d(float dtp,int nx,int ny,int nz,Array<float> an,int na,int nor,int norz,Array<float> xfall,Array<const float> dn,Array<const float> dz3d,Array<const float> dz3dinv,Array<const float> t0,Array<const float> t7,int infdo,int jslab,int its,int jts,double timesed1,double timesed2,double timesed3,double zmaxsed,double timesetvt,const float* precalculated,int ncol,int col,int selected_species) {
int ngt;
int jgs;
int ia;
int id;
constexpr int ng1=1;
int ix;
int jy;
int kz;
int ndfall;
int n;
int k;
int il;
int in;
float tmp;
float vtmax;
float dtptmp;
float dtfrac;
const float dz=200.f;
FixedStorage<float,(1-1+1)*((NZ+1)-1+1),1,1,1,(NZ+1)> db1{};
FixedStorage<float,((NZ+1)-1+1)*(1-1+1)*(1-0+1),1,(NZ+1),1,1,0,1> dtz1{};
FixedStorage<float,((NZ+1)-1+1)*(1-1+1),1,(NZ+1),1,1> dz2dinv{};
FixedStorage<float,(1-1+1)*((NZ+1)-1+1),1,1,1,(NZ+1)> db1inv{};
FixedStorage<float,(NZ-1+1)*(1-1+1),1,NZ,1,1> rhovtzx{};
FixedStorage<float,(1-1+1)*(1-1+1),1,1,1,1> xfall0{};
FixedStorage<float,((NZ+1)-1+1)*(1-1+1)*(3-1+1)*(lhab-lc+1),1,(NZ+1),1,1,1,3,lc,lhab> xvt{};
FixedStorage<float,((1+0)-((-0)+ng1)+1)*((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(1+0),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0)> tmpn{};
FixedStorage<float,((1+0)-((-0)+ng1)+1)*((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(1+0),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0)> tmpn2{};
FixedStorage<float,((1+0)-((-0)+ng1)+1)*((NZ+0)-((-0)+ng1)+1)*(lhab-lr+1),((-0)+ng1),(1+0),((-0)+ng1),(NZ+0),lr,lhab> z{};
double dummy;
double dt1;
double dt2;
double dt3;
double dt4;
int ngs;
int ngscnt;
int mgs;
int ipconc0;
FixedStorage<float,((NZ+3)-1+1)*(lhab-lv+1),1,(NZ+3),lv,lhab> qx{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-ls+1),1,(NZ+3),ls,lhab> qxw{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> cx{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> xv{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1)*(3-1+1),1,(NZ+3),lc,lhab,1,3> vtxbar{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> xmas{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> xdn{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1)*(3-1+1),1,(NZ+3),lc,lhab,1,3> xdia{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-li+1),1,(NZ+3),li,lhab> vx{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lc+1),1,(NZ+3),lc,lhab> alpha{};
FixedStorage<float,((NZ+3)-1+1)*(lhab-lr+1),1,(NZ+3),lr,lhab> zx{};
FixedStorage<bool,(1-1+1)*(lhab-(lc+1)+1),1,1,(lc+1),lhab> hasmass{};
FixedStorage<int,((NZ+3)-1+1),1,(NZ+3)> igs{};
FixedStorage<int,((NZ+3)-1+1),1,(NZ+3)> kgs{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> rho0{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> temcg{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> temg{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> rhovt{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwnc{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cinc{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> fadvisc{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cwdia{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cipmas{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cnina{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cimas{};
FixedStorage<float,((NZ+3)-1+1),1,(NZ+3)> cnostmp{};
float cimasn;
float cimasx;
int ixb;
int jyb;
int kzb;
int ixe;
int jye;
int kze;
int plo;
int phi;
bool debug_mpi=true;
// WRF:4367
; /* automatic column storage */
// WRF:4368
; /* automatic column storage */
// WRF:4369
; /* automatic column storage */
// WRF:4371
ngs = (nz+3);
// WRF:4373
; /* automatic column storage */
// WRF:4392
kzb = 1;
// WRF:4393
kze = nz;
// WRF:4395
ixb = 1;
// WRF:4396
ixe = nx;
// WRF:4399
jy = 1;
// WRF:4400
jgs = jy;
// WRF:4407
for(int _s0=1;_s0<=(nz+1);++_s0) {for(int _s1=1;_s1<=nx;++_s1) {for(int _s2=1;_s2<=3;++_s2) {for(int _s3=lc;_s3<=lhab;++_s3) {xvt(_s0,_s1,_s2,_s3) = 0.0f;}}}}
// WRF:4409
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4412
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4413
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4414
db1(ix,kz) = dn(ix,jy,kz);
// WRF:4415
db1inv(ix,kz) = lake_div(1.f,dn(ix,jy,kz));
// WRF:4416
rhovtzx(kz,ix) = sqrtf((rho00*lake_min(lake_div(1.0f,0.05f),db1inv(ix,kz))));
// WRF:4417
}
// WRF:4418
}
// WRF:4420
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4421
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4422
dtz1(kz,ix,0) = dz3dinv(ix,jy,kz);
// WRF:4423
dtz1(kz,ix,1) = (dz3dinv(ix,jy,kz)*db1inv(ix,kz));
// WRF:4424
dz2dinv(kz,ix) = dz3dinv(ix,jy,kz);
// WRF:4425
}
// WRF:4426
}
// WRF:4428
if((lzh>1)) {
// WRF:4429
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4430
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4431
an(ix,jy,kz,lzh) = lake_max(0.f,an(ix,jy,kz,lzh));
// WRF:4432
}
// WRF:4433
}
// WRF:4434
}
// WRF:4437
#pragma unroll
for(il=(lc+1);il<=lhab;il+=1) {
// WRF:4438
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4440
}
// WRF:4441
}
// WRF:4446
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4449
for(ix=ixb;ix<=ixe;ix+=1) {
// WRF:4451
dummy = 0.e0;
for(int il0=selected_species;il0<=selected_species;++il0)for(int v0=1;v0<=3;++v0)for(int k0=1;k0<=nz;++k0)xvt(k0,1,v0,il0)=precalculated[(((il0-lc)*3+v0-1)*NZ+k0-1)*ncol+col];
// WRF:4467
#pragma unroll
for(il=selected_species;il<=selected_species;il+=1) {
// WRF:4468
if((coeff_ido{}(il)==0)) {continue;}
// WRF:4476
vtmax = 0.0f;
// WRF:4478
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4481
xvt(kz,ix,1,il) = lake_min(vtmaxsed,xvt(kz,ix,1,il));
// WRF:4482
xvt(kz,ix,2,il) = lake_min(vtmaxsed,xvt(kz,ix,2,il));
// WRF:4483
xvt(kz,ix,3,il) = lake_min(vtmaxsed,xvt(kz,ix,3,il));
// WRF:4485
vtmax = lake_max(vtmax,(xvt(kz,ix,1,il)*dz2dinv(kz,ix)));
// WRF:4486
vtmax = lake_max(vtmax,(xvt(kz,ix,2,il)*dz2dinv(kz,ix)));
// WRF:4487
vtmax = lake_max(vtmax,(xvt(kz,ix,3,il)*dz2dinv(kz,ix)));
// WRF:4499
}
// WRF:4501
if((vtmax==0.0f)) {continue;}
// WRF:4505
if(((dtp*vtmax)<0.7f)) {
// WRF:4506
ndfall = 1;
// WRF:4507
} else {
// WRF:4508
if((dtp>20.0f)) {
// WRF:4509
ndfall = lake_max(2,(((int)(lake_div((dtp*vtmax),0.7f)))+1));
// WRF:4510
} else {
// WRF:4511
ndfall = (1+((int)(((dtp*vtmax)+0.301f))));
// WRF:4512
}
// WRF:4513
}
// WRF:4515
if((ndfall>1)) {
// WRF:4516
dtptmp = lake_div(dtp,((float)(ndfall)));
// WRF:4519
} else {
// WRF:4520
dtptmp = dtp;
// WRF:4521
}
// WRF:4523
dtfrac = lake_div(dtptmp,dtp);
// WRF:4526
for(n=1;n<=ndfall;n+=1) {
// WRF:4528
if(((do_accurate_sedimentation&&(n>=2))&&((n==(interval_sedi_vt*(lake_div(n,interval_sedi_vt))))))) {
// WRF:4533
dummy = 0.e0;
// WRF:4535
for(int _s0=kzb;_s0<=kze;++_s0) {for(int _s2=1;_s2<=3;++_s2) {xvt(_s0,ix,_s2,il) = 0.0f;}}
// WRF:4537
ziegfall1d(nx,ny,nz,nor,norz,na,dtp,jgs,ix,xvt,rhovtzx,an,dn,ipconc,t0,t7,cwmasn,cwmasx,cwradn,qxmin,xdnmx,xdnmn,cdx,cno,xdn0,xvmn,xvmx,ngs,qx,qxw,cx,xv,vtxbar,xmas,xdn,xdia,vx,alpha,zx,igs,kgs,rho0,temcg,temg,rhovt,cwnc,cinc,fadvisc,cwdia,cipmas,cnina,cimas,cnostmp,infdo,il);
// WRF:4548
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4550
xvt(kz,ix,1,il) = lake_min(vtmaxsed,xvt(kz,ix,1,il));
// WRF:4551
xvt(kz,ix,2,il) = lake_min(vtmaxsed,xvt(kz,ix,2,il));
// WRF:4552
xvt(kz,ix,3,il) = lake_min(vtmaxsed,xvt(kz,ix,3,il));
// WRF:4553
}
// WRF:4558
}
// WRF:4561
if((((il>=lr)&&(((infall==3)||(infall==4))))&&(coeff_ln{}(il)>0))) {
// WRF:4562
if(((((((il==lr)&&(irfall==infall))&&(lzr<1)))||(((il>=lh)&&(coeff_lz{}(il)<1))))||(((il==ls)&&(isfall==infall))))) {
// WRF:4564
calczgr1d(nx,ny,nz,nor,na,an,ixe,kze,z,db1,jgs,ipconc,coeff_dnu{}(il),il,coeff_ln{}(il),coeff_qxmin{}(il),coeff_xvmn{}(il),coeff_xvmx{}(il),coeff_lvol{}(il),coeff_xdn0{}(il),ix);
// WRF:4566
}
// WRF:4567
}
// WRF:4569
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4573
fallout1d(nx,ny,nz,nor,na,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,1,il),1,((nz)+1),1,(nx)},an,db1,il,1,xfall,dtz1,ix);
// WRF:4577
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4581
if((ldovol&&(il>=li))) {
// WRF:4582
if((coeff_lvol{}(il)>1)) {
// WRF:4583
fallout1d(nx,ny,nz,nor,na,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,1,il),1,((nz)+1),1,(nx)},an,db1,coeff_lvol{}(il),0,xfall,dtz1,ix);
// WRF:4585
}
// WRF:4586
}
// WRF:4590
if((ipconc>=6)) {
// WRF:4591
if((coeff_lz{}(il)>1)) {
// WRF:4592
fallout1d(nx,ny,nz,nor,na,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,3,il),1,((nz)+1),1,(nx)},an,db1,coeff_lz{}(il),0,xfall,dtz1,ix);
// WRF:4594
}
// WRF:4595
}
// WRF:4597
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4600
if((ipconc>0)) {
// WRF:4601
if((ipconc>=coeff_ipc{}(il))) {
// WRF:4603
if(((((infall>=2)||(((infall==0)&&(il<lh)))))&&(coeff_lz{}(il)<1))) {
// WRF:4609
if(((((infall==3)||(infall==4)))&&((((((((il==ls)&&(isfall==infall)))||(il==lh))||(il==lhl))||(il==lf))||(((il==lr)&&(irfall==infall))))))) {
// WRF:4614
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4616
tmpn2(ix,jy,kz) = z(ix,kz,il);
// WRF:4618
}
// WRF:4619
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4621
tmpn(ix,jy,kz) = an(ix,jy,kz,coeff_ln{}(il));
// WRF:4623
}
// WRF:4625
} else {
// WRF:4627
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4629
tmpn(ix,jy,kz) = an(ix,jy,kz,coeff_ln{}(il));
// WRF:4631
}
// WRF:4633
}
// WRF:4635
}
// WRF:4638
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:4640
in = 2;
// WRF:4641
if((infall==1)) {in = 1;}
// WRF:4643
fallout1d(nx,ny,nz,nor,na,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,in,il),1,((nz)+1),1,(nx)},an,db1,coeff_ln{}(il),0,xfall,dtz1,ix);
// WRF:4647
if((coeff_lz{}(il)<1)) {
// WRF:4648
if((((((infall>=2)||(infall==3)))&&(!(((infall==0)&&(il>=lh)))))&&(((il==lr)||(((il>=li)&&(il<=lhab))))))) {
// WRF:4652
for(int _s0=1;_s0<=nx;++_s0) {xfall0(_s0,jgs) = 0.0f;}
// WRF:4654
if(((((infall==3)||(infall==4)))&&((((il>=lh)||(((il==lr)&&(irfall==infall))))||(((il==ls)&&(isfall==infall))))))) {
// WRF:4657
fallout1d(nx,ny,nz,nor,1,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,3,il),1,((nz)+1),1,(nx)},tmpn2,db1,1,0,xfall0,dtz1,ix);
// WRF:4659
fallout1d(nx,ny,nz,nor,1,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,1,il),1,((nz)+1),1,(nx)},tmpn,db1,1,0,xfall0,dtz1,ix);
// WRF:4661
} else {
// WRF:4662
fallout1d(nx,ny,nz,nor,1,dtptmp,dtfrac,jgs,Array<const float>{&xvt(1,1,1,il),1,((nz)+1),1,(nx)},tmpn,db1,1,0,xfall0,dtz1,ix);
// WRF:4664
}
// WRF:4666
if(((((infall==3)||(infall==4)))&&((((((il==lr)&&(irfall==infall)))||(il>=lh))||(((il==ls)&&(isfall==infall))))))) {
// WRF:4670
calcnfromz1d(nx,ny,nz,nor,na,an,tmpn2,ixe,kze,z,db1,jgs,ipconc,coeff_dnu{}(il),il,coeff_ln{}(il),coeff_qxmin{}(il),coeff_xvmn{}(il),coeff_xvmx{}(il),tmpn,coeff_lvol{}(il),coeff_xdn0{}(il),infall,ix);
// WRF:4674
} else if((((infall==5)&&(il>=lh))||(((il==lr)&&(irfall==5))))) {
// WRF:4676
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4678
an(ix,jgs,kz,coeff_ln{}(il)) = lake_max(an(ix,jgs,kz,coeff_ln{}(il)),(0.5f*((an(ix,jgs,kz,coeff_ln{}(il))+tmpn(ix,jy,kz)))));
// WRF:4681
}
// WRF:4683
} else if(((!(((il==lr)&&(irfall==0))))&&(!(((il==ls)&&(isfall==0)))))) {
// WRF:4686
for(kz=kzb;kz<=kze;kz+=1) {
// WRF:4689
an(ix,jgs,kz,coeff_ln{}(il)) = lake_max(an(ix,jgs,kz,coeff_ln{}(il)),tmpn(ix,jy,kz));
// WRF:4692
}
// WRF:4693
}
// WRF:4694
}
// WRF:4697
}
// WRF:4698
}
// WRF:4701
}
// WRF:4704
}
// WRF:4705
}
// WRF:4707
}
// WRF:4710
; /* automatic column storage */
// WRF:4711
; /* automatic column storage */
// WRF:4712
; /* automatic column storage */
// WRF:4714
; /* automatic column storage */
// WRF:4733
return;
}
__device__ __forceinline__ void radardd02(int nx,int ny,int nz,int nor,int na,Array<const float> an,Array<const float> temk,Array<float> dbz,Array<const float> db,int nzdbz,float cnoh0t,float hwdn1t,int ipconc,int ke_diag,int iunit) {
int ngt;
constexpr int ng1=1;
int n10;
constexpr int printyn=0;
float vr;
int imapz;
int mzdist;
int vzflag;
constexpr int norz=3;
FixedStorage<float,((NZ+0)-((-0)+1)+1),((-0)+1),(NZ+0)> gz{};
float cr1;
float cr2;
float hwdnsq;
float swdnsq;
float rwdnsq;
float dhmin;
float qrmin;
float qsmin;
float qhmin;
float qhlmin;
float tfr;
float tfrh;
float zrc;
float reflectmin;
float kw_sq;
float const_ki_sn;
float const_ki_h;
float ki_sq_sn;
float ki_sq_h;
float dielf_sn;
float dielf_h;
float pi;
bool ltest;
FixedStorage<float,(1-1+1)*(NZ-1+1),1,1,1,NZ> gtmp{};
FixedStorage<float,(1-1+1)*(NZ-1+1),1,1,1,NZ> dtmp{};
float tmp;
double dtmps;
double dtmpr;
double dtmph;
double dtmphl;
double g1;
double zx;
double ze;
double x;
int i;
int j;
int k;
int ix;
int jy;
int kz;
int ihcnt;
double xcnoh;
double xcnos;
double dadh;
double dads;
double zhdryc;
double zsdryc;
double zhwetc;
double zswetc;
double dadr;
float dbzmax;
const float dbzmin=0;
float cnow;
float cnoi;
float cnoip;
float cnoir;
float cnor;
float cnos;
float cnogl;
float cnogm;
float cnogh;
float cnof;
float cnoh;
float cnohl;
float swdn;
float rwdn;
float hwdn;
float gldn;
float gmdn;
float ghdn;
float fwdn;
float hldn;
float swdn0;
float rwdnmx;
float cwdnmx;
float cidnmx;
float xidnmx;
float swdnmx;
float gldnmx;
float gmdnmx;
float ghdnmx;
float fwdnmx;
float hwdnmx;
float hldnmx;
float rwdnmn;
float cwdnmn;
float cidnmn;
float xidnmn;
float swdnmn;
float gldnmn;
float gmdnmn;
float ghdnmn;
float fwdnmn;
float hwdnmn;
float hldnmn;
float gldnsq;
float gmdnsq;
float ghdnsq;
float fwdnsq;
float hldnsq;
float dadgl;
float dadgm;
float dadgh;
float dadhl;
float dadf;
float zgldryc;
float zglwetc;
float zgmdryc;
float zgmwetc;
float zghdryc;
float zghwetc;
float zhldryc;
float zhlwetc;
float zfdryc;
float zfwetc;
float dielf_gl;
float dielf_gm;
float dielf_gh;
float dielf_hl;
float dielf_fw;
int imx;
int jmx;
int kmx;
float swdia;
float gldia;
float gmdia;
float ghdia;
float fwdia;
float hwdia;
float hldia;
float csw;
float cgl;
float cgm;
float cgh;
float cfw;
float chw;
float chl;
float xvs;
float xvgl;
float xvgm;
float xvgh;
float xvf;
float xvh;
float xvhl;
float cwc0;
int izieg;
int ice10;
const float rhos=0.1f;
float qxw;
float qxw1;
float dnsnow;
float qh;
const float cwmasn=5.23e-13f;
const float cwmasx=5.25e-10f;
const float cwradn=5.0e-6f;
FixedStorage<float,(NZ-1+1),1,NZ> cwnccn{};
float vzsnow;
float vzrain;
float vzgraupel;
float vzhail;
float ksq;
float dtp;
// WRF:8734
vzflag = 0;
// WRF:8736
izieg = 0;
// WRF:8737
ice10 = 0;
// WRF:8746
pi = (4.0f*lake_atan(1.f));
// WRF:8747
cwc0 = piinv;
// WRF:8749
cnoh = cnoh0t;
// WRF:8750
hwdn = hwdn1t;
// WRF:8752
rwdn = 1000.0f;
// WRF:8753
swdn = 100.0f;
// WRF:8755
qrmin = 1.0e-05f;
// WRF:8756
qsmin = 1.0e-06f;
// WRF:8757
qhmin = 1.0e-05f;
// WRF:8762
cnow = 1.0e+08f;
// WRF:8763
cnoi = 1.0e+08f;
// WRF:8764
cnoip = 1.0e+08f;
// WRF:8765
cnoir = 1.0e+08f;
// WRF:8766
cnor = 8.0e+06f;
// WRF:8767
cnos = 8.0e+06f;
// WRF:8768
cnogl = 4.0e+05f;
// WRF:8769
cnogm = 4.0e+05f;
// WRF:8770
cnogh = 4.0e+05f;
// WRF:8771
cnof = 4.0e+05f;
// WRF:8772
cnohl = 1.0e+03f;
// WRF:8775
imx = 1;
// WRF:8776
jmx = 1;
// WRF:8777
kmx = 1;
// WRF:8778
i = 1;
// WRF:8781
if(false) {
// WRF:8784
izieg = 1;
// WRF:8786
hwdn = hwdn1t;
// WRF:8789
cnor = coeff_cno{}(lr);
// WRF:8790
cnos = coeff_cno{}(ls);
// WRF:8791
cnoh = coeff_cno{}(lh);
// WRF:8792
qrmin = coeff_qxmin{}(lr);
// WRF:8793
qsmin = coeff_qxmin{}(ls);
// WRF:8794
qhmin = coeff_qxmin{}(lh);
// WRF:8795
if((lhl>1)) {
// WRF:8796
cnohl = coeff_cno{}(lhl);
// WRF:8797
qhlmin = coeff_qxmin{}(lhl);
// WRF:8798
}
// WRF:8800
} else if(true) {
// WRF:8802
izieg = 1;
// WRF:8804
swdn0 = swdn;
// WRF:8806
cnor = coeff_cno{}(lr);
// WRF:8807
cnos = coeff_cno{}(ls);
// WRF:8808
cnoh = coeff_cno{}(lh);
// WRF:8810
qrmin = coeff_qxmin{}(lr);
// WRF:8811
qsmin = coeff_qxmin{}(ls);
// WRF:8812
qhmin = coeff_qxmin{}(lh);
// WRF:8813
if((lhl>1)) {
// WRF:8814
cnohl = coeff_cno{}(lhl);
// WRF:8815
qhlmin = coeff_qxmin{}(lhl);
// WRF:8816
}
// WRF:8820
}
// WRF:8900
rwdnmx = 1000.0f;
// WRF:8901
cwdnmx = 1000.0f;
// WRF:8902
cidnmx = 917.0f;
// WRF:8903
xidnmx = 917.0f;
// WRF:8904
swdnmx = 200.0f;
// WRF:8905
gldnmx = 400.0f;
// WRF:8906
gmdnmx = 600.0f;
// WRF:8907
ghdnmx = 800.0f;
// WRF:8908
fwdnmx = 900.0f;
// WRF:8909
hwdnmx = 900.0f;
// WRF:8910
hldnmx = 900.0f;
// WRF:8912
rwdnmn = 1000.0f;
// WRF:8913
cwdnmn = 1000.0f;
// WRF:8914
xidnmn = 001.0f;
// WRF:8915
cidnmn = 001.0f;
// WRF:8916
swdnmn = 001.0f;
// WRF:8917
gldnmn = 200.0f;
// WRF:8918
gmdnmn = 400.0f;
// WRF:8919
ghdnmn = 600.0f;
// WRF:8920
fwdnmn = 700.0f;
// WRF:8921
hwdnmn = 700.0f;
// WRF:8922
hldnmn = 900.0f;
// WRF:8925
gldn = ((0.5f)*((gldnmn+gldnmx)));
// WRF:8926
gmdn = ((0.5f)*((gmdnmn+gmdnmx)));
// WRF:8927
ghdn = ((0.5f)*((ghdnmn+ghdnmx)));
// WRF:8928
fwdn = ((0.5f)*((fwdnmn+fwdnmx)));
// WRF:8929
hldn = ((0.5f)*((hldnmn+hldnmx)));
// WRF:8932
cr1 = 7.2e+20f;
// WRF:8933
cr2 = 7.295e+19f;
// WRF:8934
hwdnsq = lake_pow(hwdn,2);
// WRF:8935
swdnsq = lake_pow(swdn,2);
// WRF:8936
rwdnsq = lake_pow(rwdn,2);
// WRF:8938
gldnsq = lake_pow(gldn,2);
// WRF:8939
gmdnsq = lake_pow(gmdn,2);
// WRF:8940
ghdnsq = lake_pow(ghdn,2);
// WRF:8941
fwdnsq = lake_pow(fwdn,2);
// WRF:8942
hldnsq = lake_pow(hldn,2);
// WRF:8944
dhmin = 0.005f;
// WRF:8945
tfr = 273.16f;
// WRF:8946
tfrh = (tfr-8.0f);
// WRF:8947
zrc = (cr1*cnor);
// WRF:8948
reflectmin = 0.0f;
// WRF:8949
kw_sq = 0.93f;
// WRF:8950
dbzmax = dbzmin;
// WRF:8952
ihcnt = 0;
// WRF:8962
const_ki_sn = (0.5f-(lake_div(((0.5f-0.46f)),((917.f-220.f)))*((swdn-220.f))));
// WRF:8963
const_ki_h = (0.5f-(lake_div(((0.5f-0.46f)),((917.f-220.f)))*((hwdn-220.f))));
// WRF:8964
ki_sq_sn = ((lake_div(swdnsq,rwdnsq))*lake_pow(const_ki_sn,2));
// WRF:8965
ki_sq_h = ((lake_div(hwdnsq,rwdnsq))*lake_pow(const_ki_h,2));
// WRF:8966
dielf_sn = lake_div(ki_sq_sn,kw_sq);
// WRF:8967
dielf_h = lake_div(ki_sq_h,kw_sq);
// WRF:8974
dielf_sn = lake_div(((lake_div(swdnsq,rwdnsq))*.21f),kw_sq);
// WRF:8975
dielf_h = lake_div(((lake_div(hwdnsq,rwdnsq))*.21f),kw_sq);
// WRF:8977
dielf_gl = lake_div(((lake_div(gldnsq,rwdnsq))*.21f),kw_sq);
// WRF:8978
dielf_gm = lake_div(((lake_div(gmdnsq,rwdnsq))*.21f),kw_sq);
// WRF:8979
dielf_gh = lake_div(((lake_div(ghdnsq,rwdnsq))*.21f),kw_sq);
// WRF:8980
dielf_hl = lake_div(((lake_div(hldnsq,rwdnsq))*.21f),kw_sq);
// WRF:8981
dielf_fw = lake_div(((lake_div(fwdnsq,rwdnsq))*.21f),kw_sq);
// WRF:9007
for(jy=1;jy<=1;jy+=1) {
// WRF:9009
for(kz=1;kz<=ke_diag;kz+=1) {
// WRF:9011
for(ix=1;ix<=nx;ix+=1) {
// WRF:9012
dbz(ix,jy,kz) = 0.0f;
// WRF:9014
vzsnow = 0.0f;
// WRF:9015
vzrain = 0.0f;
// WRF:9016
vzgraupel = 0.0f;
// WRF:9017
vzhail = 0.0f;
// WRF:9019
dtmph = 0.0f;
// WRF:9020
dtmps = 0.0f;
// WRF:9021
dtmphl = 0.0f;
// WRF:9022
dtmpr = 0.0f;
// WRF:9023
dadr = lake_pow((lake_div(db(ix,jy,kz),(((pi*rwdn)*cnor)))),(0.25f));
// WRF:9028
dtmp(ix,kz) = 0.0f;
// WRF:9029
gtmp(ix,kz) = 0.0f;
// WRF:9030
if((an(ix,jy,kz,lr)>=qrmin)) {
// WRF:9031
if((ipconc<=2)) {
// WRF:9032
gtmp(ix,kz) = (dadr*lake_pow(an(ix,jy,kz,lr),(0.25f)));
// WRF:9033
dtmp(ix,kz) = (zrc*lake_pow(gtmp(ix,kz),7));
// WRF:9034
} else if((lzr>1)) {
// WRF:9035
dtmp(ix,kz) = (1e18f*an(ix,jy,kz,lzr));
// WRF:9036
} else if((an(ix,jy,kz,lnr)>1.e-3f)) {
// WRF:9037
if((imurain==3)) {
// WRF:9038
vr = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lr)),((1000.f*an(ix,jy,kz,lnr))));
// WRF:9039
dtmp(ix,kz) = lake_div((((3.6e18f*((rnu+2.f)))*an(ix,jy,kz,lnr))*lake_pow(vr,2)),((rnu+1.f)));
// WRF:9040
} else {
// WRF:9041
g1 = lake_div(((((6.0f+alphar))*((5.0f+alphar)))*((4.0f+alphar))),(((((3.0f+alphar))*((2.0f+alphar)))*((1.0f+alphar)))));
// WRF:9042
zx = lake_div((g1*lake_pow(((db(ix,jy,kz)*an(ix,jy,kz,lr))),2)),an(ix,jy,kz,lnr));
// WRF:9043
ze = ((1.e18f*zx)*lake_pow((lake_div(6.f,((pi*1000.f)))),2));
// WRF:9044
dtmp(ix,kz) = ze;
// WRF:9045
}
// WRF:9046
}
// WRF:9047
dtmpr = dtmp(ix,kz);
// WRF:9048
}
// WRF:9056
if((lhab>lr)) {
// WRF:9083
xcnoh = cnoh;
// WRF:9084
xcnos = cnos;
// WRF:9102
if((ls>1)) {
// WRF:9104
if((lvs>1)) {
// WRF:9105
if((an(ix,jy,kz,lvs)>0.0f)) {
// WRF:9106
swdn = lake_div((db(ix,jy,kz)*an(ix,jy,kz,ls)),an(ix,jy,kz,lvs));
// WRF:9107
swdn = lake_min(300.f,lake_max(100.f,swdn));
// WRF:9108
} else {
// WRF:9109
swdn = swdn0;
// WRF:9110
}
// WRF:9112
}
// WRF:9114
if((ipconc>=5)) {
// WRF:9116
xvs = lake_div((db(ix,jy,kz)*an(ix,jy,kz,ls)),((swdn*lake_max(1.0e-3f,an(ix,jy,kz,lns)))));
// WRF:9118
if(((xvs<xvsmn)||(xvs>xvsmx))) {
// WRF:9119
xvs = lake_min(xvsmx,lake_max(xvsmn,xvs));
// WRF:9120
csw = lake_div((db(ix,jy,kz)*an(ix,jy,kz,ls)),((xvs*swdn)));
// WRF:9121
}
// WRF:9123
swdia = lake_pow(((xvs*cwc0)),(lake_div(1.f,3.f)));
// WRF:9124
xcnos = lake_div((an(ix,jy,kz,ls)*db(ix,jy,kz)),(((xvs*swdn)*swdia)));
// WRF:9126
}
// WRF:9127
}
// WRF:9141
if((lh>1)) {
// WRF:9143
if((lvh>1)) {
// WRF:9144
if((an(ix,jy,kz,lvh)>0.0f)) {
// WRF:9145
hwdn = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),an(ix,jy,kz,lvh));
// WRF:9146
hwdn = lake_min(900.f,lake_max(hdnmn,hwdn));
// WRF:9147
} else {
// WRF:9148
hwdn = 500.f;
// WRF:9149
}
// WRF:9150
} else {
// WRF:9151
hwdn = hwdn1t;
// WRF:9152
}
// WRF:9154
if((ipconc>=5)) {
// WRF:9156
xvh = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),((hwdn*lake_max(1.0e-3f,an(ix,jy,kz,lnh)))));
// WRF:9158
if(((xvh<xvhmn)||(xvh>xvhmx))) {
// WRF:9159
xvh = lake_min(xvhmx,lake_max(xvhmn,xvh));
// WRF:9160
chw = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),((xvh*hwdn)));
// WRF:9161
}
// WRF:9163
hwdia = lake_pow(((xvh*cwc0)),(lake_div(1.f,3.f)));
// WRF:9164
xcnoh = lake_div((an(ix,jy,kz,lh)*db(ix,jy,kz)),(((xvh*hwdn)*hwdia)));
// WRF:9166
}
// WRF:9168
}
// WRF:9170
dadh = 0.0f;
// WRF:9171
dadhl = 0.0f;
// WRF:9172
dads = 0.0f;
// WRF:9173
if((xcnoh>0.0f)) {
// WRF:9174
dadh = lake_pow((lake_div(db(ix,jy,kz),(((pi*hwdn)*xcnoh)))),(.25f));
// WRF:9175
zhdryc = lake_div(((0.224f*cr2)*lake_pow((lake_div(db(ix,jy,kz),rwdn)),2)),xcnoh);
// WRF:9179
} else {
// WRF:9180
dadh = 0.0f;
// WRF:9181
zhdryc = 0.0f;
// WRF:9182
}
// WRF:9184
if((xcnos>0.0f)) {
// WRF:9185
dads = lake_pow((lake_div(db(ix,jy,kz),(((pi*swdn)*xcnos)))),(.25f));
// WRF:9186
zsdryc = lake_div(((0.224f*cr2)*lake_pow((lake_div(db(ix,jy,kz),rwdn)),2)),xcnos);
// WRF:9187
} else {
// WRF:9188
dads = 0.0f;
// WRF:9189
zsdryc = 0.0f;
// WRF:9190
}
// WRF:9191
zhwetc = zhdryc;
// WRF:9192
zswetc = zsdryc;
// WRF:9196
if((ls>1)) {
// WRF:9198
gtmp(ix,kz) = 0.0f;
// WRF:9199
qxw = 0.0f;
// WRF:9200
qxw1 = 0.0f;
// WRF:9201
dtmps = 0.0f;
// WRF:9202
if((an(ix,jy,kz,ls)>=qsmin)) {
// WRF:9203
if((ipconc>=4)) {
// WRF:9205
if((lsw>1)) {
// WRF:9206
qxw = an(ix,jy,kz,lsw);
// WRF:9207
qxw1 = 0.0f;
// WRF:9208
} else if(((((((iusewetsnow==1)||(iusewetsnow==3)))&&(temk(ix,jy,kz)>(tfr+1.f)))&&(an(ix,jy,kz,ls)>an(ix,jy,kz,lr)))&&(an(ix,jy,kz,lr)>qsmin))) {
// WRF:9210
qxw = lake_min((0.5f*an(ix,jy,kz,ls)),an(ix,jy,kz,lr));
// WRF:9211
qxw1 = qxw;
// WRF:9212
}
// WRF:9214
vr = xvs;
// WRF:9217
ksq = 0.189f;
// WRF:9218
if((an(ix,jy,kz,lns)>1.e-7f)) {
// WRF:9220
if(((qxw>qsmin)||(iusewetsnow>=2))) {
// WRF:9223
gtmp(ix,kz) = (lake_div((((3.6e18f*((snu+2.f)))*(((0.224f*((an(ix,jy,kz,ls)+qxw1)))+(0.776f*qxw))))*((an(ix,jy,kz,ls)+qxw1))),(((an(ix,jy,kz,lns)*((snu+1.f)))*lake_pow(rwdn,2))))*lake_pow(db(ix,jy,kz),2));
// WRF:9226
} else {
// WRF:9228
dnsnow = (0.0346159f*sqrtf(lake_div(an(ix,jy,kz,lns),((an(ix,jy,kz,ls)*db(ix,jy,kz))))));
// WRF:9229
if((true||(dnsnow<900.f))) {
// WRF:9230
gtmp(ix,kz) = lake_div(((((((1.e18f*323.3226f)*lake_pow(0.106214f,2))*(((ksq*an(ix,jy,kz,ls))+(((1.f-ksq))*qxw))))*an(ix,jy,kz,ls))*lake_pow(db(ix,jy,kz),2))*gsnow73),((((an(ix,jy,kz,lns)*lake_pow((917.f),2))*gsnow1)*lake_pow(((1.0f+snu)),(lake_div(4.f,3.f))))));
// WRF:9233
} else {
// WRF:9234
gtmp(ix,kz) = (lake_div((((((lake_div(36.f,lake_pow(pi,2)))*1.e18f)*((snu+2.f)))*(((0.224f*((an(ix,jy,kz,ls)+qxw1)))+(0.776f*qxw))))*((an(ix,jy,kz,ls)+qxw1))),(((an(ix,jy,kz,lns)*((snu+1.f)))*lake_pow(rwdn,2))))*lake_pow(db(ix,jy,kz),2));
// WRF:9236
}
// WRF:9238
}
// WRF:9240
}
// WRF:9244
dtmps = gtmp(ix,kz);
// WRF:9245
dtmp(ix,kz) = (dtmp(ix,kz)+gtmp(ix,kz));
// WRF:9246
} else {
// WRF:9247
gtmp(ix,kz) = (dads*lake_pow(an(ix,jy,kz,ls),(0.25f)));
// WRF:9249
if((gtmp(ix,kz)>0.0f)) {
// WRF:9250
dtmps = lake_div((zsdryc*lake_pow(an(ix,jy,kz,ls),2)),gtmp(ix,kz));
// WRF:9251
if((temk(ix,jy,kz)<tfr)) {
// WRF:9252
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zsdryc*lake_pow(an(ix,jy,kz,ls),2)),gtmp(ix,kz)));
// WRF:9254
} else {
// WRF:9255
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zswetc*lake_pow(an(ix,jy,kz,ls),2)),gtmp(ix,kz)));
// WRF:9257
}
// WRF:9258
}
// WRF:9259
}
// WRF:9261
}
// WRF:9263
}
// WRF:9269
if(((li>1)&&(idbzci!=0))) {
// WRF:9271
if(((idbzci==1)&&(lni>0))) {
// WRF:9273
if(((an(ix,jy,kz,li)>coeff_qxmin{}(li))&&(an(ix,jy,kz,lni)>1.0f))) {
// WRF:9274
vr = lake_div((db(ix,jy,kz)*an(ix,jy,kz,li)),((900.f*an(ix,jy,kz,lni))));
// WRF:9275
dtmp(ix,kz) = (dtmp(ix,kz)+(lake_div(((((0.224f*3.6e18f)*((cinu+2.f)))*an(ix,jy,kz,lni))*lake_pow(vr,2)),((cinu+1.f)))*lake_pow((lake_div(900.f,1000.f)),2)));
// WRF:9277
}
// WRF:9279
} else if((idbzci==2)) {
// WRF:9283
gtmp(ix,kz) = 0.0f;
// WRF:9284
if((an(ix,jy,kz,li)>=0.1e-3f)) {
// WRF:9285
gtmp(ix,kz) = lake_min(1.0f,((1.e3f*(an(ix,jy,kz,li)))*db(ix,jy,kz)));
// WRF:9286
dtmp(ix,kz) = (dtmp(ix,kz)+(750.0f*lake_pow((gtmp(ix,kz)),1.98f)));
// WRF:9287
}
// WRF:9289
}
// WRF:9291
}
// WRF:9296
if((lh>1)) {
// WRF:9297
gtmp(ix,kz) = 0.0f;
// WRF:9298
dtmph = 0.0f;
// WRF:9299
qxw = 0.0f;
// WRF:9301
if(((izieg>=1)&&(ipconc>=5))) {
// WRF:9303
ltest = false;
// WRF:9304
if((lzh>1)) {
// WRF:9305
if((((an(ix,jy,kz,lzh)>0.0f)&&(an(ix,jy,kz,lh)>qhmin))&&(an(ix,jy,kz,lnh)>=cxmin))) {ltest = true;}
// WRF:9307
}
// WRF:9309
if((ltest||(((an(ix,jy,kz,lh)>=qhmin)&&(an(ix,jy,kz,lnh)>=cxmin))))) {
// WRF:9311
if((lvh>1)) {
// WRF:9313
if((an(ix,jy,kz,lvh)>0.0f)) {
// WRF:9314
hwdn = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),an(ix,jy,kz,lvh));
// WRF:9315
hwdn = lake_min(900.f,lake_max(100.f,hwdn));
// WRF:9316
} else {
// WRF:9317
hwdn = 500.f;
// WRF:9318
}
// WRF:9320
}
// WRF:9322
chw = an(ix,jy,kz,lnh);
// WRF:9323
if((chw>0.0f)) {
// WRF:9324
xvh = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),((hwdn*lake_max(1.0e-3f,chw))));
// WRF:9325
if(((xvh<xvhmn)||(xvh>xvhmx))) {
// WRF:9326
xvh = lake_min(xvhmx,lake_max(xvhmn,xvh));
// WRF:9327
chw = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lh)),((xvh*hwdn)));
// WRF:9328
}
// WRF:9330
qh = an(ix,jy,kz,lh);
// WRF:9332
if((lhw>1)) {
// WRF:9333
if((iusewetgraupel==1)) {
// WRF:9334
qxw = an(ix,jy,kz,lhw);
// WRF:9335
} else if((iusewetgraupel==2)) {
// WRF:9336
if((hwdn<300.f)) {
// WRF:9337
qxw = an(ix,jy,kz,lhw);
// WRF:9338
}
// WRF:9339
}
// WRF:9340
} else if((iusewetgraupel==3)) {
// WRF:9341
if((((hwdn<300.f)&&(temk(ix,jy,kz)>tfr))&&(an(ix,jy,kz,lr)>qhmin))) {
// WRF:9342
qxw = lake_min(an(ix,jy,kz,lh),an(ix,jy,kz,lr));
// WRF:9343
qh = (qh+qxw);
// WRF:9344
}
// WRF:9345
} else if(((((iusewetgraupel==4)&&(temk(ix,jy,kz)>(tfr+0.25f)))&&(an(ix,jy,kz,lh)>an(ix,jy,kz,lr)))&&(an(ix,jy,kz,lr)>qhmin))) {
// WRF:9347
qxw = lake_min((0.5f*an(ix,jy,kz,lh)),an(ix,jy,kz,lr));
// WRF:9348
qh = (qh+qxw);
// WRF:9350
}
// WRF:9352
if((lzh>1)) {
// WRF:9353
x = lake_div((((0.224f*qh)+(0.776f*qxw))),an(ix,jy,kz,lh));
// WRF:9354
dtmph = (((1.e18f*x)*an(ix,jy,kz,lzh))*lake_pow((lake_div(hwdn,rwdn)),2));
// WRF:9355
dtmp(ix,kz) = (dtmp(ix,kz)+dtmph);
// WRF:9356
} else {
// WRF:9357
g1 = lake_div(((((6.0f+alphah))*((5.0f+alphah)))*((4.0f+alphah))),(((((3.0f+alphah))*((2.0f+alphah)))*((1.0f+alphah)))));
// WRF:9360
zx = lake_div((((g1*lake_pow(db(ix,jy,kz),2))*(((0.224f*qh)+(0.776f*qxw))))*qh),chw);
// WRF:9361
ze = ((1.e18f*zx)*lake_pow((lake_div(6.f,((pi*1000.f)))),2));
// WRF:9362
dtmp(ix,kz) = (dtmp(ix,kz)+ze);
// WRF:9363
dtmph = ze;
// WRF:9364
}
// WRF:9366
}
// WRF:9369
}
// WRF:9371
} else {
// WRF:9373
dtmph = 0.0f;
// WRF:9375
if((an(ix,jy,kz,lh)>=qhmin)) {
// WRF:9376
gtmp(ix,kz) = (dadh*lake_pow(an(ix,jy,kz,lh),(0.25f)));
// WRF:9377
if((gtmp(ix,kz)>0.0f)) {
// WRF:9378
dtmph = lake_div((zhdryc*lake_pow(an(ix,jy,kz,lh),2)),gtmp(ix,kz));
// WRF:9379
if((temk(ix,jy,kz)<tfr)) {
// WRF:9380
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zhdryc*lake_pow(an(ix,jy,kz,lh),2)),gtmp(ix,kz)));
// WRF:9382
} else {
// WRF:9384
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zhdryc*lake_pow(an(ix,jy,kz,lh),2)),gtmp(ix,kz)));
// WRF:9391
}
// WRF:9392
}
// WRF:9393
}
// WRF:9397
}
// WRF:9400
}
// WRF:9402
}
// WRF:9405
if(((izieg>=1)&&(lhl>1))) {
// WRF:9407
hldn = 900.0f;
// WRF:9408
gtmp(ix,kz) = 0.0f;
// WRF:9409
dtmphl = 0.0f;
// WRF:9410
qxw = 0.0f;
// WRF:9413
if((lvhl>1)) {
// WRF:9414
if((an(ix,jy,kz,lvhl)>0.0f)) {
// WRF:9415
hldn = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lhl)),an(ix,jy,kz,lvhl));
// WRF:9416
hldn = lake_min(900.f,lake_max(300.f,hldn));
// WRF:9417
} else {
// WRF:9418
hldn = 900.f;
// WRF:9419
}
// WRF:9420
} else {
// WRF:9421
hldn = rho_qhl;
// WRF:9422
}
// WRF:9425
if((ipconc>=5)) {
// WRF:9427
ltest = false;
// WRF:9428
if((lzhl>1)) {
// WRF:9429
if((((an(ix,jy,kz,lzhl)>0.0f)&&(an(ix,jy,kz,lhl)>qhlmin))&&(an(ix,jy,kz,lnhl)>0.0f))) {ltest = true;}
// WRF:9431
}
// WRF:9433
if((ltest||(((an(ix,jy,kz,lhl)>=qhlmin)&&(an(ix,jy,kz,lnhl)>0.f))))) {
// WRF:9434
chl = an(ix,jy,kz,lnhl);
// WRF:9435
if((chl>0.0f)) {
// WRF:9436
xvhl = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lhl)),((hldn*lake_max(1.0e-9f,an(ix,jy,kz,lnhl)))));
// WRF:9438
if(((xvhl<xvhlmn)||(xvhl>xvhlmx))) {
// WRF:9439
xvhl = lake_min(xvhlmx,lake_max(xvhlmn,xvhl));
// WRF:9440
chl = lake_div((db(ix,jy,kz)*an(ix,jy,kz,lhl)),((xvhl*hldn)));
// WRF:9442
}
// WRF:9444
if((lhlw>1)) {
// WRF:9445
if((iusewethail==1)) {
// WRF:9446
qxw = an(ix,jy,kz,lhlw);
// WRF:9447
} else if((iusewethail==2)) {
// WRF:9448
if((hldn<300.f)) {
// WRF:9449
qxw = an(ix,jy,kz,lhlw);
// WRF:9450
}
// WRF:9451
}
// WRF:9452
}
// WRF:9454
if((lzhl>1)) {
// WRF:9455
x = lake_div((((0.224f*an(ix,jy,kz,lhl))+(0.776f*qxw))),an(ix,jy,kz,lhl));
// WRF:9456
dtmphl = (((1.e18f*x)*an(ix,jy,kz,lzhl))*lake_pow((lake_div(hldn,rwdn)),2));
// WRF:9457
dtmp(ix,kz) = (dtmp(ix,kz)+dtmphl);
// WRF:9458
} else {
// WRF:9460
g1 = lake_div(((((6.0f+alphahl))*((5.0f+alphahl)))*((4.0f+alphahl))),(((((3.0f+alphahl))*((2.0f+alphahl)))*((1.0f+alphahl)))));
// WRF:9461
zx = lake_div((((g1*lake_pow(db(ix,jy,kz),2))*(((0.224f*an(ix,jy,kz,lhl))+(0.776f*qxw))))*an(ix,jy,kz,lhl)),chl);
// WRF:9463
ze = ((1.e18f*zx)*lake_pow((lake_div(6.f,((pi*1000.f)))),2));
// WRF:9464
dtmp(ix,kz) = (dtmp(ix,kz)+ze);
// WRF:9465
dtmphl = ze;
// WRF:9467
}
// WRF:9468
}
// WRF:9470
}
// WRF:9473
} else {
// WRF:9476
if((an(ix,jy,kz,lhl)>=qhlmin)) {
// WRF:9477
dadhl = lake_pow((lake_div(db(ix,jy,kz),(((pi*hldn)*cnohl)))),(.25f));
// WRF:9478
gtmp(ix,kz) = (dadhl*lake_pow(an(ix,jy,kz,lhl),(0.25f)));
// WRF:9479
if((gtmp(ix,kz)>0.0f)) {
// WRF:9481
zhldryc = lake_div(((0.224f*cr2)*lake_pow((lake_div(db(ix,jy,kz),rwdn)),2)),cnohl);
// WRF:9483
dtmphl = lake_div((zhldryc*lake_pow(an(ix,jy,kz,lhl),2)),gtmp(ix,kz));
// WRF:9485
if((temk(ix,jy,kz)<tfr)) {
// WRF:9486
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zhldryc*lake_pow(an(ix,jy,kz,lhl),2)),gtmp(ix,kz)));
// WRF:9488
} else {
// WRF:9490
dtmp(ix,kz) = (dtmp(ix,kz)+lake_div((zhldryc*lake_pow(an(ix,jy,kz,lhl),2)),gtmp(ix,kz)));
// WRF:9497
}
// WRF:9498
}
// WRF:9500
}
// WRF:9502
}
// WRF:9505
}
// WRF:9509
if((dtmp(ix,kz)>0.0f)) {
// WRF:9510
dbz(ix,jy,kz) = lake_max(dbzmin,(10.0f*nssl1_log10f(dtmp(ix,kz))));
// WRF:9512
if((dbz(ix,jy,kz)>dbzmax)) {
// WRF:9513
dbzmax = lake_max(dbzmax,dbz(ix,jy,kz));
// WRF:9514
imx = ix;
// WRF:9515
jmx = jy;
// WRF:9516
kmx = kz;
// WRF:9517
}
// WRF:9518
} else {
// WRF:9519
dbz(ix,jy,kz) = dbzmin;
// WRF:9520
if(((lh>1)&&(lhl>1))) {
// WRF:9521
if((an(ix,jy,kz,lh)>1.0e-3f)) {
// WRF:9522
; /* native diagnostic only */
// WRF:9523
; /* native diagnostic only */
// WRF:9525
if(((lzh>1)&&(lzhl>1))) {; /* native diagnostic only */}
// WRF:9526
}
// WRF:9527
}
// WRF:9528
}
// WRF:9538
if(((!(dtmp(ix,kz)<1.e30f))||(dbz(ix,jy,kz)>190.0f))) {
// WRF:9541
; /* native diagnostic only */
// WRF:9542
; /* native diagnostic only */
// WRF:9543
; /* native diagnostic only */
// WRF:9544
; /* native diagnostic only */
// WRF:9545
; /* native diagnostic only */
// WRF:9546
; /* native diagnostic only */
// WRF:9547
; /* native diagnostic only */
// WRF:9548
; /* native diagnostic only */
// WRF:9549
if((lhl>1)) {; /* native diagnostic only */}
// WRF:9550
if((ipconc>=3)) {; /* native diagnostic only */}
// WRF:9551
if((lzr>1)) {; /* native diagnostic only */}
// WRF:9552
if((ipconc>=5)) {
// WRF:9553
; /* native diagnostic only */
// WRF:9554
if((lhl>1)) {; /* native diagnostic only */}
// WRF:9555
if((lzhl>1)) {
// WRF:9556
; /* native diagnostic only */
// WRF:9557
; /* native diagnostic only */
// WRF:9558
; /* native diagnostic only */
// WRF:9559
}
// WRF:9560
}
// WRF:9561
; /* native diagnostic only */
// WRF:9562
; /* native diagnostic only */
// WRF:9563
; /* native diagnostic only */
// WRF:9564
; /* native diagnostic only */
// WRF:9565
if((!(((dbz(ix,jy,kz)>(-100))&&(dbz(ix,jy,kz)<200))))) {
// WRF:9566
; /* native diagnostic only */
// WRF:9568
}
// WRF:9569
}
// WRF:9572
}
// WRF:9573
}
// WRF:9574
}
// WRF:9580
if((printyn==1)) {
// WRF:9582
; /* native diagnostic only */
// WRF:9583
; /* native diagnostic only */
// WRF:9585
if((lh>1)) {
// WRF:9586
; /* native diagnostic only */
// WRF:9587
; /* native diagnostic only */
// WRF:9588
; /* native diagnostic only */
// WRF:9589
if((lhl>1)) {; /* native diagnostic only */}
// WRF:9590
}
// WRF:9593
}
// WRF:9596
return;
}
__device__ __forceinline__ void nssl_2mom_gs(int nx,int ny,int nz,int na,int jyslab,int nor,int norz,float dtp,Array<const float> gz,Array<float> t0,Array<const float> t1,Array<const float> t2,Array<const float> t3,Array<const float> t4,Array<const float> t5,Array<const float> t6,Array<const float> t7,Array<const float> t8,Array<float> t9,Array<float> an,Array<const float> dn,Array<float> p2,Array<const float> pn,Array<const float> w,int iunit,Array<const float> t00,Array<const float> t77,float ventr,float ventc,float c1sw,int jgs,Array<const int> ido,Array<const float> xdnmx,Array<const float> xdnmn,Array<const float> cdx,Array<const float> xdn0,Array<const float> tmp3d,Array<const float> tkediss,Array<const float> thproc,int numproc,float dx1,float dy1,int ngs,double timevtcalc,Array<const float> axtra,bool io_flag,bool has_wetscav,Array<float> rainprod2d,Array<float> evapprod2d,Array<float> alpha2d,Array<const float> elec,int its,int ids,int ide,int jds,int jde) {
constexpr int ng1=1;
int nba;
int nv;
int istag;
int jstag;
int kstag;
int iwrite;
float dx;
float dy;
float dz;
int itile;
int jtile;
int ktile;
int ixbeg;
int jybeg;
int ixend;
int jyend;
int kzend;
int kzbeg;
int nxend;
int nyend;
int nzend;
int nzbeg;
int my_rank=0;
constexpr int myprock=1;
constexpr int nprock=1;
const float tfrdry=0x1.e64ccc0000000p+7f;
FixedStorage<bool,(lhab-lc+1),lc,lhab> lrescalelow{};
float galpharaut;
float xvbarmax;
float qvex;
constexpr int iraincv=1;
constexpr int icgxconv=1;
float ffrz;
float ffrzh=1.0f;
float qcitmp;
float cirdiatmp;
float ccwtmp;
float ccitmp;
float cpqc;
float cpci;
float cpqc0;
float cpci0;
float scfac;
double dp1;
double frac;
double frach;
double xvfrz;
double xvbiggsnow;
double dpt1;
double dpt2;
constexpr bool gammacheck=false;
int luindex;
double tmpgam;
constexpr bool usegamxinfcnu=false;
constexpr bool usegamxinf=false;
constexpr bool usegamxinf2=false;
constexpr bool usegamxinf3=false;
float vtmax;
int n;
int ndfall;
double chgneg;
double chgpos;
double sctot;
float temgtmp;
FixedStorage<float,((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(NZ+0)> pb{};
FixedStorage<float,((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(NZ+0)> pinit{};
float qimax;
float xni0;
float roqi0;
float dv;
float dtptmp;
constexpr int itest=1;
constexpr int nidx=10;
constexpr int id1=1;
constexpr int jd1=1;
constexpr int kd1=1;
int ierr;
int iend;
int ix;
int kz;
int il;
int ic;
int ir;
int icp1;
int irp1;
int ip1;
int jp1;
int kp1;
int jy;
int i;
int j;
int k;
int i1;
int kzb;
int kze;
float slope1;
float slope2;
float x1;
float x2;
float x3;
const float eps=0x1.79ca100000000p-67f;
const float eps2=0x1.4f8b580000000p-17f;
float temele;
float trev;
bool ldovol;
bool ishail;
bool ltest;
bool wtest;
constexpr bool alp0flag=false;
constexpr int mu=1;
constexpr int mv=2;
constexpr int mw=3;
constexpr int mqcw=21;
constexpr int mqxw=21;
constexpr int mtem=21;
constexpr int mrho=5;
constexpr int mtim=6;
const float xftim=0x1.99999a0000000p-5f;
const float xftimi=0x1.4000000000000p+4f;
const float yftim=0x1.0000000000000p+0f;
const float xftem=0x1.0000000000000p-1f;
const float yftem=0x1.0000000000000p+0f;
const float xfqcw=0x1.f400000000000p+10f;
const float yfqcw=0x1.0000000000000p+0f;
const float xfqxw=0x1.f400000000000p+10f;
const float yfqxw=0x1.0000000000000p+0f;
const float dtfac=0x1.0000000000000p+0f;
FixedStorage<float,(lqmx-lc+1),lc,lqmx> delqnxa{};
FixedStorage<float,(lqmx-lc+1),lc,lqmx> delqxxa{};
int nxmpb;
int nzmpb;
int nxz;

int numgs;
constexpr int ngsz=500;
constexpr int ntt=300;
FixedStorage<float,(1-1+1),1,1> dvmgs{};
int ngscnt;
FixedStorage<int,(1-1+1),1,1> igs{};
FixedStorage<int,(1-1+1),1,1> kgs{};
FixedStorage<int,(1-1+1),1,1> kgsp{};
FixedStorage<int,(1-1+1),1,1> kgsm{};
FixedStorage<int,(1-1+1),1,1> kgsm2{};
constexpr int ncuse=0;
FixedStorage<int,(1-1+1),1,1> il0{};
FixedStorage<int,(1-1+1),1,1> il5{};
FixedStorage<int,(1-1+1),1,1> il2{};
FixedStorage<int,(1-1+1),1,1> il3{};
float tdtol;
float temsav;
float tfrcbw;
float tfrcbi;
const float thnuc=0x1.d64ccc0000000p+7f;
FixedStorage<float,(1-1+1),1,1> fimt1{};
FixedStorage<float,(1-1+1),1,1> fimta{};
FixedStorage<float,(1-1+1),1,1> fimt2{};
float xcwmas;
FixedStorage<float,(1-1+1),1,1> ccnc{};
FixedStorage<float,(1-1+1),1,1> ccin{};
FixedStorage<float,(1-1+1),1,1> cina{};
FixedStorage<float,(1-1+1),1,1> ccna{};
FixedStorage<float,(1-1+1),1,1> cwnccn{};
const float sscb=0x1.0000000000000p+1f;
constexpr int idecss=1;
constexpr int iba=1;
constexpr int ifilt=0;
float temp1;
float temp2;
float mwat;
float mice;
float dice;
float mwshed;
float fwmax;
float fw;
float mwcrit;
float massfactor;
float tmpdiam;
const float shedalp=0x1.8000000000000p+1f;
FixedStorage<float,(1-1+1),1,1> ssmax{};
float ssmx;
float dnnet;
float dqnet;
float bfnu;
const float bfnu0=0x1.8000020000000p+2f;
float bfnu1;
float volb;
double t2s;
double xdp;
FixedStorage<double,(1-1+1),1,1> xl2p{};
FixedStorage<double,(1-1+1),1,1> rb{};
const float aa1=0x1.0c4d0a0000000p+53f;
const float aa2=0x1.6940000000000p+12f;
const float cexs=0x1.99999a0000000p-4f;
const float cecs=0x1.0000000000000p-1f;
const float rvt=0x1.a9fbe80000000p-4f;
const float kfrag=0x1.0c6f7a0000000p-20f;
const float mfrag=0x1.b7cdfe0000000p-34f;
FixedStorage<double,(1-1+1),1,1> cautn{};
FixedStorage<double,(1-1+1),1,1> rh{};
FixedStorage<double,(1-1+1),1,1> nh{};
float ex1;
float ft;
FixedStorage<float,(1-1+1),1,1> rhoinv{};
FixedStorage<double,(1-1+1),1,1> ec0{};
float ac1;
float bc;
float taus;
float c1;
float d1;
float e1;
float f1;
float p380;
float tmp;
float tmp1;
float tmp2;
float tmp3;
float tmp4;
float tmp5;
float tmp6;
float temp3;
float flim;
float dw;
float dwr;
double tmpz;
double tmpzmlt;
float ratio;
float delx;
float dely;
float dbigg;
float volt;
float chgtmp;
float fac;
float mixedphasefac;
float x;
float y;
float y2;
float del;
float r;
float rtmp;
float alpr;
double vent1;
double vent2;
double g1palp;
double g4palp;
double g1palpinf;
double g4palpinf;
float fqt;
float bs;
float v1;
float v2;
float d1r;
float d1i;
float d1s;
float e1i;
const float vr1mm=0x1.1fda020000000p-31f;
const float vr3mm=0x1.e5bfe40000000p-27f;
const float vr4p5mm=0x1.99d9e80000000p-25f;
float vmlt;
float vshd;
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> vshdgs{};
FixedStorage<float,(lhab-lc+1),lc,lhab> maxmassfac{};
const float rhosm=0x1.f400000000000p+8f;
int nc;
float dtcon;
float dtcon1;
float dtcon2;
float delta;
int ltemq1;
int ltemq1m;
float dqv;
float qv1;
float ss1;
float ss2;
float qvs1;
float dqvs;
float dtemp;
float dt1;
float ssi1;
float ssi2;
float dqvi;
float dqvis;
float dqvii;
float qis1;
float dqvr;
float dqc;
float dqr;
float dqi;
float dqs;
float qv1m;
float qvs1m;
float ss1m;
float ssi1m;
float qis1m;
float cwmastmp;
float dcloud;
float dcloud2;
FixedStorage<float,(1-1+1),1,1> cn{};
double xvc;
double xvr;
float mwfac;
float rwmasn;
float rwmasx;
const float vgra=0x1.1fda060000000p-31f;
float vfrz;
float d;
float dold;
float denom;
float denominv;
float vth;
double h1;
double h2;
double h3;
double h4;
double denomdp;
double denominvdp;
float r1;
float qevap;
float vr;
float nrx;
float chw;
float g1;
float qr;
float z;
float z1;
float rdi;
float alp;
float xnutmp;
float xnuc;
float g1r;
float rd1;
float rdia;
float rmas;
float snowmeltmass=0;
const float rimedens=0x1.f400000000000p+8f;
const float raero=0x1.421f600000000p-22f;
const float kaero=0x1.613d320000000p-8f;
const float kb=0x1.0b10f40000000p-76f;
FixedStorage<float,(1-1+1),1,1> knud{};
FixedStorage<float,(1-1+1),1,1> knuda{};
FixedStorage<float,(1-1+1),1,1> gtp{};
FixedStorage<float,(1-1+1),1,1> dfar{};
FixedStorage<float,(1-1+1),1,1> fn1{};
FixedStorage<float,(1-1+1),1,1> fn2{};
FixedStorage<float,(1-1+1),1,1> fnft{};
FixedStorage<float,(1-1+1),1,1> ccia{};
FixedStorage<float,(1-1+1),1,1> ctfzbd{};
FixedStorage<float,(1-1+1),1,1> ctfzth{};
FixedStorage<float,(1-1+1),1,1> ctfzdi{};
float ni;
float nis;
float nr;
float d0;
FixedStorage<float,(1-1+1),1,1> dqvcnd{};
FixedStorage<float,(1-1+1),1,1> dqwv{};
FixedStorage<float,(1-1+1),1,1> dqcw{};
FixedStorage<float,(1-1+1),1,1> dqci{};
FixedStorage<float,(1-1+1),1,1> dqcitmp{};
FixedStorage<float,(1-1+1),1,1> dqwvtmp{};
FixedStorage<float,(1-1+1),1,1> tempc{};
FixedStorage<float,(1-1+1),1,1> temg{};
FixedStorage<float,(1-1+1),1,1> temcg{};
FixedStorage<float,(1-1+1),1,1> theta{};
FixedStorage<float,(1-1+1),1,1> qvap{};
FixedStorage<float,(1-1+1),1,1> temgkm1{};
FixedStorage<float,(1-1+1),1,1> temgkm2{};
FixedStorage<float,(1-1+1),1,1> temgx{};
FixedStorage<float,(1-1+1),1,1> temcgx{};
FixedStorage<float,(1-1+1),1,1> qvs{};
FixedStorage<float,(1-1+1),1,1> qis{};
FixedStorage<float,(1-1+1),1,1> qss{};
FixedStorage<float,(1-1+1),1,1> pqs{};
FixedStorage<float,(1-1+1),1,1> elv{};
FixedStorage<float,(1-1+1),1,1> elf{};
FixedStorage<float,(1-1+1),1,1> els{};
FixedStorage<float,(1-1+1),1,1> tsqr{};
FixedStorage<float,(1-1+1),1,1> ssi{};
FixedStorage<float,(1-1+1),1,1> ssw{};
FixedStorage<float,(1-1+1),1,1> ssi0{};
FixedStorage<float,(1-1+1),1,1> qcwtmp{};
float qtmp;
FixedStorage<float,(1-1+1),1,1> qtot{};
FixedStorage<float,(1-1+1),1,1> qcond{};
float ctmp;
float sctmp;
float cimasn;
float cimasx;
float ccimx;
float pid4;
float cs;
float ds;
float gf7;
float gf6;
float gf5;
float gf4;
float gf3;
float gf2;
float gf1;
float gcnup1;
float gcnup2;
float gf73rds;
float gf83rds;
float gamice73fac;
float gamsnow73fac;
float gf43rds;
float gf53rds;
float aradcw;
float bradcw;
float cradcw;
float dradcw;
float cwrad;
float rwrad;
const float rwradmn=0x1.a36e2e0000000p-15f;
float dh0;
FixedStorage<float,(1-1+1),1,1> dg0{};
FixedStorage<float,(1-1+1),1,1> df0{};
FixedStorage<float,(1-1+1),1,1> dhwet{};
FixedStorage<float,(1-1+1),1,1> dhlwet{};
FixedStorage<float,(1-1+1),1,1> dfwet{};
const float clionpmx=0x1.dcd6500000000p+29f;
const float clionnmx=0x1.dcd6500000000p+29f;
FixedStorage<float,(1-1+1),1,1> fwet1{};
FixedStorage<float,(1-1+1),1,1> fwet2{};
FixedStorage<float,(1-1+1),1,1> fmlt1{};
FixedStorage<float,(1-1+1),1,1> fmlt2{};
FixedStorage<float,(1-1+1),1,1> fmlt1e{};
FixedStorage<float,(1-1+1),1,1> fvds{};
FixedStorage<float,(1-1+1),1,1> fvce{};
FixedStorage<float,(1-1+1),1,1> fiinit{};
FixedStorage<float,(1-1+1),1,1> fvent{};
FixedStorage<float,(1-1+1),1,1> fraci{};
FixedStorage<float,(1-1+1),1,1> fracl{};
FixedStorage<float,(1-1+1),1,1> fai{};
FixedStorage<float,(1-1+1),1,1> fav{};
FixedStorage<float,(1-1+1),1,1> fbi{};
FixedStorage<float,(1-1+1),1,1> fbv{};
FixedStorage<float,(1-1+1),1,1> felv{};
FixedStorage<float,(1-1+1),1,1> fels{};
FixedStorage<float,(1-1+1),1,1> felf{};
FixedStorage<float,(1-1+1),1,1> felvcp{};
FixedStorage<float,(1-1+1),1,1> felscp{};
FixedStorage<float,(1-1+1),1,1> felfcp{};
FixedStorage<float,(1-1+1),1,1> felvpi{};
FixedStorage<float,(1-1+1),1,1> felspi{};
FixedStorage<float,(1-1+1),1,1> felfpi{};
FixedStorage<float,(1-1+1),1,1> felvs{};
FixedStorage<float,(1-1+1),1,1> felss{};
FixedStorage<float,(1-1+1),1,1> fwvdf{};
FixedStorage<float,(1-1+1),1,1> ftka{};
FixedStorage<float,(1-1+1),1,1> fthdf{};
FixedStorage<float,(1-1+1),1,1> fadvisc{};
FixedStorage<float,(1-1+1),1,1> fakvisc{};
FixedStorage<float,(1-1+1),1,1> fci{};
FixedStorage<float,(1-1+1),1,1> fcw{};
FixedStorage<float,(1-1+1),1,1> fschm{};
FixedStorage<float,(1-1+1),1,1> fpndl{};
FixedStorage<float,(1-1+1),1,1> fgamw{};
FixedStorage<float,(1-1+1),1,1> fgams{};
FixedStorage<float,(1-1+1),1,1> fcqv1{};
FixedStorage<float,(1-1+1),1,1> fcqv2{};
FixedStorage<float,(1-1+1),1,1> fcc3{};
float cvm;
float cpm;
float rmm;
const float cpv=0x1.d740000000000p+10f;
FixedStorage<float,(1-1+1),1,1> fcci{};
FixedStorage<float,(1-1+1),1,1> fcip{};
FixedStorage<float,(1-1+1),1,1> sfm1{};
FixedStorage<float,(1-1+1),1,1> sfm2{};
FixedStorage<float,(1-1+1),1,1> gfm1{};
FixedStorage<float,(1-1+1),1,1> gfm2{};
FixedStorage<float,(1-1+1),1,1> ffm1{};
FixedStorage<float,(1-1+1),1,1> ffm2{};
FixedStorage<float,(1-1+1),1,1> hfm1{};
FixedStorage<float,(1-1+1),1,1> hfm2{};
FixedStorage<bool,(1-1+1),1,1> wetsfc{};
FixedStorage<bool,(1-1+1),1,1> wetsfchl{};
FixedStorage<bool,(1-1+1),1,1> wetsfcf{};
FixedStorage<bool,(1-1+1),1,1> wetgrowth{};
FixedStorage<bool,(1-1+1),1,1> wetgrowthhl{};
FixedStorage<bool,(1-1+1),1,1> wetgrowthf{};
FixedStorage<float,(1-1+1),1,1> qitmp{};
FixedStorage<float,(1-1+1),1,1> qistmp{};
FixedStorage<float,(1-1+1),1,1> rzxh{};
FixedStorage<float,(1-1+1),1,1> rzxhl{};
FixedStorage<float,(1-1+1),1,1> rzxhlh{};
FixedStorage<float,(1-1+1),1,1> rzxhlf{};
FixedStorage<float,(1-1+1),1,1> rzxs{};
FixedStorage<float,(1-1+1),1,1> rzxf{};
FixedStorage<float,(1-1+1),1,1> cdh{};
FixedStorage<float,(1-1+1),1,1> cdhl{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> axx{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> bxx{};
FixedStorage<float,(1-1+1),1,1> vt2ave{};
FixedStorage<float,(1-1+1),1,1> qcwresv{};
FixedStorage<float,(1-1+1),1,1> ccwresv{};
FixedStorage<float,(1-1+1)*(6-1+1),1,1,1,6> lfsave{};
FixedStorage<float,(1-1+1)*(lhab-lv+1),1,1,lv,lhab> qx{};
FixedStorage<float,(1-1+1)*(lhab-ls+1),1,1,ls,lhab> qxw{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> qxwlg{};
FixedStorage<float,(1-1+1)*(lhab-lh+1),1,1,lh,lhab> chxf{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cxmxd{};
FixedStorage<float,(1-1+1)*(lhab-lv+1),1,1,lv,lhab> qxmxd{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> scx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xv{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(3-1+1),1,1,lc,lhab,1,3> vtxbar{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xmas{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xdn{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xdntmp{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cdxgs{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(3-1+1),1,1,lc,lhab,1,3> xdia{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> vtwtdia{};
FixedStorage<float,(1-1+1)*(lhab-ls+1),1,1,ls,lhab> rarx{};
FixedStorage<float,(1-1+1)*(lhab-li+1),1,1,li,lhab> vx{};
FixedStorage<float,(1-1+1)*(lhab-li+1),1,1,li,lhab> rimdn{};
FixedStorage<float,(1-1+1)*(lhab-li+1),1,1,li,lhab> raindn{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> alpha{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(lhab-lc+1),1,1,lc,lhab,lc,lhab> dab0lh{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(lhab-lc+1),1,1,lc,lhab,lc,lhab> dab1lh{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> zx{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> zxmxd{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> g1x{};
FixedStorage<float,(1-1+1),1,1> qsimxdep{};
FixedStorage<float,(1-1+1),1,1> qsimxsub{};
constexpr bool dosublimationfix=true;
FixedStorage<float,(1-1+1),1,1> qrtmp{};
FixedStorage<float,(1-1+1),1,1> qvtmp{};
FixedStorage<float,(1-1+1),1,1> qctmp{};
float felvcptmp;
float felscptmp;
float qsstmp;
float thetatmp;
float thetaptmp;
float temcgtmp;
float qvaptmp;
float qvstmp;
float qisstmp;
float qvptmp;
float qitmp1;
float qctmp1;
float galphrout;
FixedStorage<float,(1-1+1),1,1> ventrx{};
FixedStorage<float,(1-1+1),1,1> ventrxn{};
float g1shr;
float alphashr;
float g1mlr;
float alphamlr;
float g1smlr;
float alphasmlr;
float massfacshr;
float massfacmlr;
float qhgt8mm;
float qhwgt8mm;
float qhgt10mm;
float qhgt20mm;
float fwmhtmp;
const float fwmhtmptem=-0x1.e000000000000p+3f;
const float d1t=0x1.0dcc4e0000000p-7f;
const float srasheym=0x1.1c779a0000000p-3f;
FixedStorage<float,(1-1+1),1,1> swvent{};
FixedStorage<float,(1-1+1),1,1> hwvent{};
FixedStorage<float,(1-1+1),1,1> rwvent{};
FixedStorage<float,(1-1+1),1,1> hlvent{};
FixedStorage<float,(1-1+1),1,1> hwventy{};
FixedStorage<float,(1-1+1),1,1> hlventy{};
FixedStorage<float,(1-1+1),1,1> rwventz{};
float hxventtmp;
FixedStorage<float,(1-1+1),1,1> hlventinc{};
FixedStorage<float,(1-1+1),1,1> hwventinc{};
constexpr int ndiam=10;
int numdiam;
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> hwvent0{};
float hlvent0;
float hwvent1;
float hlvent1;
float hwvent2;
float hlvent2;
float gama0;
float gamb0;
float gama1;
float gamb1;
float gama2;
float gamb2;
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> mltdiam{};
float mltmass0inv;
float mltmass1inv;
float mltmass2inv;
float mltmass1cgs;
float mltmass2cgs;
float mltmass3inv;
float mltmass3cgs;
float qhmlr0;
float qhmlr05;
float qhmlr1;
float qhmlr2;
float qhmlr3;
float qhmlr12;
float qhmlr23;
float qhlmlr0;
float qhlmlr05;
float qhlmlr1;
float qhlmlr2;
float qhlmlr3;
float qhlmlr12;
float qhlmlr23;
float qxd1;
float cxd1;
float zxd1;
float qxd05;
float cxd05;
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> qxd{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> cxd{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> qhml{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> qhml0{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> dqxd{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> dcxd{};
FixedStorage<float,((ndiam+4)-1+1),1,(ndiam+4)> dqhml{};
FixedStorage<float,(1-1+1),1,1> civent{};
FixedStorage<float,(1-1+1),1,1> isvent{};
FixedStorage<float,(1-1+1),1,1> xmascw{};
float dnmx;
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xdiamxmas{};
FixedStorage<float,(1-1+1),1,1> cilen{};
FixedStorage<float,(1-1+1),1,1> rwcap{};
FixedStorage<float,(1-1+1),1,1> swcap{};
FixedStorage<float,(1-1+1),1,1> hwcap{};
FixedStorage<float,(1-1+1),1,1> hlcap{};
FixedStorage<float,(1-1+1),1,1> cicap{};
FixedStorage<float,(1-1+1),1,1> iscap{};
FixedStorage<float,(1-1+1),1,1> qvimxd{};
FixedStorage<float,(1-1+1),1,1> qimxd{};
FixedStorage<float,(1-1+1),1,1> qismxd{};
FixedStorage<float,(1-1+1),1,1> qcmxd{};
FixedStorage<float,(1-1+1),1,1> qrmxd{};
FixedStorage<float,(1-1+1),1,1> qsmxd{};
FixedStorage<float,(1-1+1),1,1> qhmxd{};
FixedStorage<float,(1-1+1),1,1> qhlmxd{};
FixedStorage<float,(1-1+1),1,1> cimxd{};
FixedStorage<float,(1-1+1),1,1> ccmxd{};
FixedStorage<float,(1-1+1),1,1> crmxd{};
FixedStorage<float,(1-1+1),1,1> csmxd{};
FixedStorage<float,(1-1+1),1,1> chmxd{};
FixedStorage<float,(1-1+1),1,1> cionpmxd{};
FixedStorage<float,(1-1+1),1,1> cionnmxd{};
FixedStorage<float,(1-1+1),1,1> clionpmxd{};
FixedStorage<float,(1-1+1),1,1> clionnmxd{};
FixedStorage<float,(1-1+1),1,1> chmul1{};
FixedStorage<float,(1-1+1),1,1> chlmul1{};
FixedStorage<float,(1-1+1),1,1> csmul1{};
FixedStorage<float,(1-1+1),1,1> csmul{};
FixedStorage<float,(1-1+1),1,1> qhmul1{};
FixedStorage<float,(1-1+1),1,1> qhlmul1{};
FixedStorage<float,(1-1+1),1,1> qsmul1{};
FixedStorage<float,(1-1+1),1,1> qsmul{};
FixedStorage<float,(1-1+1),1,1> csplinter{};
FixedStorage<float,(1-1+1),1,1> qsplinter{};
FixedStorage<float,(1-1+1),1,1> csplinter2{};
FixedStorage<float,(1-1+1),1,1> qsplinter2{};
FixedStorage<float,(1-1+1),1,1> chlcnh{};
FixedStorage<float,(1-1+1),1,1> vhlcnh{};
FixedStorage<float,(1-1+1),1,1> vhlcnhl{};
FixedStorage<float,(1-1+1),1,1> chlcnhhl{};
FixedStorage<float,(1-1+1),1,1> cracif{};
FixedStorage<float,(1-1+1),1,1> ciacrf{};
FixedStorage<float,(1-1+1),1,1> cracr{};
FixedStorage<float,(1-1+1),1,1> ciint{};
FixedStorage<float,(1-1+1),1,1> crfrz{};
FixedStorage<float,(1-1+1),1,1> crfrzf{};
FixedStorage<float,(1-1+1),1,1> crfrzs{};
FixedStorage<float,(1-1+1),1,1> cicint{};
FixedStorage<float,(1-1+1),1,1> cipint{};
FixedStorage<float,(1-1+1),1,1> ciacw{};
FixedStorage<float,(1-1+1),1,1> cwacii{};
FixedStorage<float,(1-1+1),1,1> ciacr{};
FixedStorage<float,(1-1+1),1,1> craci{};
FixedStorage<float,(1-1+1),1,1> csacw{};
FixedStorage<float,(1-1+1),1,1> csacr{};
FixedStorage<float,(1-1+1),1,1> csaci{};
FixedStorage<float,(1-1+1),1,1> csacs{};
FixedStorage<float,(1-1+1),1,1> cracw{};
FixedStorage<float,(1-1+1),1,1> chacw{};
FixedStorage<float,(1-1+1),1,1> chacr{};
FixedStorage<float,(1-1+1),1,1> chlacw{};
FixedStorage<float,(1-1+1),1,1> chaci{};
FixedStorage<float,(1-1+1),1,1> chacs{};
FixedStorage<float,(1-1+1),1,1> chlacr{};
FixedStorage<float,(1-1+1),1,1> chlaci{};
FixedStorage<float,(1-1+1),1,1> chlacs{};
FixedStorage<float,(1-1+1),1,1> crcnw{};
FixedStorage<float,(1-1+1),1,1> cidpv{};
FixedStorage<float,(1-1+1),1,1> cisbv{};
FixedStorage<float,(1-1+1),1,1> cisdpv{};
FixedStorage<float,(1-1+1),1,1> cissbv{};
FixedStorage<float,(1-1+1),1,1> cimlr{};
FixedStorage<float,(1-1+1),1,1> cismlr{};
FixedStorage<float,(1-1+1),1,1> chlsbv{};
FixedStorage<float,(1-1+1),1,1> chldpv{};
FixedStorage<float,(1-1+1),1,1> chlmlr{};
FixedStorage<float,(1-1+1),1,1> chlmlrr{};
FixedStorage<float,(1-1+1),1,1> chlfmlr{};
FixedStorage<float,(1-1+1),1,1> chlshr{};
FixedStorage<float,(1-1+1),1,1> chlshrr{};
FixedStorage<float,(1-1+1),1,1> chdpv{};
FixedStorage<float,(1-1+1),1,1> chsbv{};
FixedStorage<float,(1-1+1),1,1> chmlr{};
FixedStorage<float,(1-1+1),1,1> chcev{};
FixedStorage<float,(1-1+1),1,1> chmlrr{};
FixedStorage<float,(1-1+1),1,1> chshr{};
FixedStorage<float,(1-1+1),1,1> chshrr{};
FixedStorage<float,(1-1+1),1,1> csdpv{};
FixedStorage<float,(1-1+1),1,1> cssbv{};
FixedStorage<float,(1-1+1),1,1> csmlr{};
FixedStorage<float,(1-1+1),1,1> csmlrr{};
FixedStorage<float,(1-1+1),1,1> cscev{};
FixedStorage<float,(1-1+1),1,1> csshr{};
FixedStorage<float,(1-1+1),1,1> csshrr{};
FixedStorage<float,(1-1+1),1,1> crcev{};
FixedStorage<float,(1-1+1),1,1> crshr{};
FixedStorage<float,(1-1+1),1,1> cwshw{};
FixedStorage<float,(1-1+1),1,1> qwshw{};
FixedStorage<float,(1-1+1),1,1> qrcnw{};
FixedStorage<float,(1-1+1),1,1> qwcnr{};
FixedStorage<float,(1-1+1),1,1> zrcnw{};
FixedStorage<float,(1-1+1),1,1> zracr{};
FixedStorage<float,(1-1+1),1,1> zracw{};
FixedStorage<float,(1-1+1),1,1> zrcev{};
FixedStorage<float,(1-1+1),1,1> qracw{};
FixedStorage<float,(1-1+1),1,1> qiacw{};
FixedStorage<float,(1-1+1),1,1> qsacw{};
FixedStorage<float,(1-1+1),1,1> qhacw{};
FixedStorage<float,(1-1+1),1,1> qhlacw{};
float qxacwtmp;
float qxacrtmp;
float qxacitmp;
float qxacstmp;
FixedStorage<float,(1-1+1),1,1> vhacw{};
FixedStorage<float,(1-1+1),1,1> vsacw{};
FixedStorage<float,(1-1+1),1,1> vhlacw{};
FixedStorage<float,(1-1+1),1,1> vhlacr{};
FixedStorage<float,(1-1+1),1,1> qfcev{};
FixedStorage<float,(1-1+1),1,1> qfmul1{};
FixedStorage<float,(1-1+1),1,1> cfmul1{};
FixedStorage<float,(1-1+1),1,1> qsacws{};
FixedStorage<float,(1-1+1),1,1> qsacr{};
FixedStorage<float,(1-1+1),1,1> qracs{};
FixedStorage<float,(1-1+1),1,1> qhacr{};
FixedStorage<float,(1-1+1),1,1> qhacrmlr{};
FixedStorage<float,(1-1+1),1,1> qhacwmlr{};
FixedStorage<float,(1-1+1),1,1> vhacr{};
FixedStorage<float,(1-1+1),1,1> zhacr{};
FixedStorage<float,(1-1+1),1,1> zhacrf{};
FixedStorage<float,(1-1+1),1,1> zrach{};
FixedStorage<float,(1-1+1),1,1> zrachl{};
FixedStorage<float,(1-1+1),1,1> qiacr{};
FixedStorage<float,(1-1+1),1,1> qraci{};
FixedStorage<float,(1-1+1),1,1> ziacr{};
FixedStorage<float,(1-1+1),1,1> qracif{};
FixedStorage<float,(1-1+1),1,1> qiacrf{};
FixedStorage<float,(1-1+1),1,1> qiacrs{};
FixedStorage<float,(1-1+1),1,1> ciacrs{};
FixedStorage<float,(1-1+1),1,1> qhlacr{};
FixedStorage<float,(1-1+1),1,1> qhlacrmlr{};
FixedStorage<float,(1-1+1),1,1> qhlacwmlr{};
FixedStorage<float,(1-1+1),1,1> qsacrs{};
FixedStorage<float,(1-1+1),1,1> qsaci{};
FixedStorage<float,(1-1+1),1,1> qsacis{};
FixedStorage<float,(1-1+1),1,1> qhaci{};
FixedStorage<float,(1-1+1),1,1> qhacs{};
FixedStorage<float,(1-1+1),1,1> qhacis{};
FixedStorage<float,(1-1+1),1,1> chacis{};
FixedStorage<float,(1-1+1),1,1> chacis0{};
FixedStorage<float,(1-1+1),1,1> csaci0{};
FixedStorage<float,(1-1+1),1,1> chaci0{};
FixedStorage<float,(1-1+1),1,1> chacs0{};
FixedStorage<float,(1-1+1),1,1> chlaci0{};
FixedStorage<float,(1-1+1),1,1> chlacis{};
FixedStorage<float,(1-1+1),1,1> chlacis0{};
FixedStorage<float,(1-1+1),1,1> chlacs0{};
FixedStorage<float,(1-1+1),1,1> qsaci0{};
FixedStorage<float,(1-1+1),1,1> qsacis0{};
FixedStorage<float,(1-1+1),1,1> qhaci0{};
FixedStorage<float,(1-1+1),1,1> qhacis0{};
FixedStorage<float,(1-1+1),1,1> qhacs0{};
FixedStorage<float,(1-1+1),1,1> qhlaci0{};
FixedStorage<float,(1-1+1),1,1> qhlacis0{};
FixedStorage<float,(1-1+1),1,1> qhlacs0{};
FixedStorage<float,(1-1+1),1,1> qhlaci{};
FixedStorage<float,(1-1+1),1,1> qhlacis{};
FixedStorage<float,(1-1+1),1,1> qhlacs{};
FixedStorage<float,(1-1+1),1,1> qrfrz{};
FixedStorage<float,(1-1+1),1,1> zrfrz{};
FixedStorage<float,(1-1+1),1,1> zrfrzf{};
FixedStorage<float,(1-1+1),1,1> zrfrzs{};
FixedStorage<float,(1-1+1),1,1> ziacrf{};
FixedStorage<float,(1-1+1),1,1> zhcnsh{};
FixedStorage<float,(1-1+1),1,1> zhcnih{};
FixedStorage<float,(1-1+1),1,1> zhacw{};
FixedStorage<float,(1-1+1),1,1> zhacs{};
FixedStorage<float,(1-1+1),1,1> zhaci{};
FixedStorage<float,(1-1+1),1,1> zhmlr{};
FixedStorage<float,(1-1+1),1,1> zhdsv{};
FixedStorage<float,(1-1+1),1,1> zhsbv{};
FixedStorage<float,(1-1+1),1,1> zhlcnh{};
FixedStorage<float,(1-1+1),1,1> zhshr{};
FixedStorage<float,(1-1+1),1,1> zfacw{};
FixedStorage<float,(1-1+1),1,1> zfacs{};
FixedStorage<float,(1-1+1),1,1> zfaci{};
FixedStorage<float,(1-1+1),1,1> zfmlr{};
FixedStorage<float,(1-1+1),1,1> zfdsv{};
FixedStorage<float,(1-1+1),1,1> zfsbv{};
FixedStorage<float,(1-1+1),1,1> zhlcnf{};
FixedStorage<float,(1-1+1),1,1> zfshr{};
FixedStorage<float,(1-1+1),1,1> zfshrr{};
float zhmlrtmp;
float zhmlr0inf;
float zhlmlr0inf;
FixedStorage<float,(1-1+1),1,1> zhmlrr{};
FixedStorage<float,(1-1+1),1,1> zhlmlrr{};
FixedStorage<float,(1-1+1),1,1> zhshrr{};
FixedStorage<float,(1-1+1),1,1> zhlshrr{};
FixedStorage<float,(1-1+1),1,1> zfmlrr{};
FixedStorage<float,(1-1+1),1,1> zsmlrr{};
FixedStorage<float,(1-1+1),1,1> zsshr{};
FixedStorage<float,(1-1+1),1,1> zsshrr{};
FixedStorage<float,(1-1+1),1,1> zhcns{};
FixedStorage<float,(1-1+1),1,1> zhcni{};
FixedStorage<float,(1-1+1),1,1> zhwdn{};
FixedStorage<float,(1-1+1),1,1> zfwdn{};
FixedStorage<float,(1-1+1),1,1> zhldn{};
FixedStorage<float,(1-1+1),1,1> zhlacw{};
FixedStorage<float,(1-1+1),1,1> zhlacs{};
FixedStorage<float,(1-1+1),1,1> zhlacr{};
FixedStorage<float,(1-1+1),1,1> zhlmlr{};
FixedStorage<float,(1-1+1),1,1> zhldsv{};
FixedStorage<float,(1-1+1),1,1> zhlsbv{};
FixedStorage<float,(1-1+1),1,1> zhlshr{};
FixedStorage<float,(1-1+1),1,1> vrfrzf{};
FixedStorage<float,(1-1+1),1,1> viacrf{};
FixedStorage<float,(1-1+1),1,1> qrfrzs{};
FixedStorage<float,(1-1+1),1,1> qrfrzf{};
FixedStorage<float,(1-1+1),1,1> qwfrz{};
FixedStorage<float,(1-1+1),1,1> qwctfz{};
FixedStorage<float,(1-1+1),1,1> cwfrz{};
FixedStorage<float,(1-1+1),1,1> cwctfz{};
FixedStorage<float,(1-1+1),1,1> qwfrzis{};
FixedStorage<float,(1-1+1),1,1> qwctfzis{};
FixedStorage<float,(1-1+1),1,1> cwfrzis{};
FixedStorage<float,(1-1+1),1,1> cwctfzis{};
FixedStorage<float,(1-1+1),1,1> qwfrzc{};
FixedStorage<float,(1-1+1),1,1> qwctfzc{};
FixedStorage<float,(1-1+1),1,1> cwfrzc{};
FixedStorage<float,(1-1+1),1,1> cwctfzc{};
FixedStorage<float,(1-1+1),1,1> qwfrzp{};
FixedStorage<float,(1-1+1),1,1> qwctfzp{};
FixedStorage<float,(1-1+1),1,1> cwfrzp{};
FixedStorage<float,(1-1+1),1,1> cwctfzp{};
FixedStorage<float,(1-1+1),1,1> xcolmn{};
FixedStorage<float,(1-1+1),1,1> xplate{};
FixedStorage<float,(1-1+1),1,1> ciihr{};
FixedStorage<float,(1-1+1),1,1> qiihr{};
FixedStorage<float,(1-1+1),1,1> cicichr{};
FixedStorage<float,(1-1+1),1,1> qicichr{};
FixedStorage<float,(1-1+1),1,1> cipiphr{};
FixedStorage<float,(1-1+1),1,1> qipiphr{};
FixedStorage<float,(1-1+1),1,1> qscni{};
FixedStorage<float,(1-1+1),1,1> cscni{};
FixedStorage<float,(1-1+1),1,1> cscnis{};
FixedStorage<float,(1-1+1),1,1> qscnvi{};
FixedStorage<float,(1-1+1),1,1> cscnvi{};
FixedStorage<float,(1-1+1),1,1> cscnvis{};
FixedStorage<float,(1-1+1),1,1> qhcns{};
FixedStorage<float,(1-1+1),1,1> chcns{};
FixedStorage<float,(1-1+1),1,1> chcnsh{};
FixedStorage<float,(1-1+1),1,1> vhcns{};
FixedStorage<float,(1-1+1),1,1> qscnh{};
FixedStorage<float,(1-1+1),1,1> cscnh{};
FixedStorage<float,(1-1+1),1,1> vscnh{};
FixedStorage<float,(1-1+1),1,1> qhcni{};
FixedStorage<float,(1-1+1),1,1> chcni{};
FixedStorage<float,(1-1+1),1,1> chcnih{};
FixedStorage<float,(1-1+1),1,1> vhcni{};
FixedStorage<float,(1-1+1),1,1> qiint{};
FixedStorage<float,(1-1+1),1,1> qipipnt{};
FixedStorage<float,(1-1+1),1,1> qicicnt{};
FixedStorage<float,(1-1+1),1,1> cninm{};
FixedStorage<float,(1-1+1),1,1> cnina{};
FixedStorage<float,(1-1+1),1,1> cninp{};
FixedStorage<float,(1-1+1),1,1> wvel{};
FixedStorage<float,(1-1+1),1,1> wvelkm1{};
FixedStorage<float,(1-1+1),1,1> tke{};
FixedStorage<float,(1-1+1),1,1> uvel{};
FixedStorage<float,(1-1+1),1,1> vvel{};
FixedStorage<float,(1-1+1),1,1> qidpv{};
FixedStorage<float,(1-1+1),1,1> qisbv{};
FixedStorage<float,(1-1+1),1,1> qimlr{};
FixedStorage<float,(1-1+1),1,1> qidsv{};
FixedStorage<float,(1-1+1),1,1> qisdsv{};
FixedStorage<float,(1-1+1),1,1> qidsvp{};
FixedStorage<float,(1-1+1),1,1> qismlr{};
FixedStorage<float,(1-1+1),1,1> qhldpv{};
FixedStorage<float,(1-1+1),1,1> qhlsbv{};
FixedStorage<float,(1-1+1),1,1> qhlmlr{};
FixedStorage<float,(1-1+1),1,1> qhldsv{};
FixedStorage<float,(1-1+1),1,1> qhlmlrsave{};
FixedStorage<float,(1-1+1),1,1> qhlwet{};
FixedStorage<float,(1-1+1),1,1> qhldry{};
FixedStorage<float,(1-1+1),1,1> qhlshr{};
float qxwettmp;
FixedStorage<float,(1-1+1),1,1> qrfz{};
FixedStorage<float,(1-1+1),1,1> qsfz{};
FixedStorage<float,(1-1+1),1,1> qhfz{};
FixedStorage<float,(1-1+1),1,1> qhlfz{};
FixedStorage<float,(1-1+1),1,1> qffz{};
FixedStorage<float,(1-1+1),1,1> qhdpv{};
FixedStorage<float,(1-1+1),1,1> qhsbv{};
FixedStorage<float,(1-1+1),1,1> qhmlr{};
FixedStorage<float,(1-1+1),1,1> qhdsv{};
FixedStorage<float,(1-1+1),1,1> qhcev{};
FixedStorage<float,(1-1+1),1,1> qhcndv{};
FixedStorage<float,(1-1+1),1,1> qhevv{};
FixedStorage<float,(1-1+1),1,1> qhlcev{};
FixedStorage<float,(1-1+1),1,1> chlcev{};
FixedStorage<float,(1-1+1),1,1> qhwet{};
FixedStorage<float,(1-1+1),1,1> qhdry{};
FixedStorage<float,(1-1+1),1,1> qhshr{};
FixedStorage<float,(1-1+1),1,1> qhshrp{};
FixedStorage<float,(1-1+1),1,1> qhshh{};
FixedStorage<float,(1-1+1),1,1> qhmlh{};
FixedStorage<float,(1-1+1),1,1> qhfzh{};
FixedStorage<float,(1-1+1),1,1> qffzf{};
FixedStorage<float,(1-1+1),1,1> qhlfzhl{};
FixedStorage<float,(1-1+1),1,1> qhmlrlg{};
FixedStorage<float,(1-1+1),1,1> qhlmlrlg{};
FixedStorage<float,(1-1+1),1,1> qhfzhlg{};
FixedStorage<float,(1-1+1),1,1> qhlfzhllg{};
FixedStorage<float,(1-1+1),1,1> qhlcevlg{};
FixedStorage<float,(1-1+1),1,1> chlcevlg{};
FixedStorage<float,(1-1+1),1,1> qhcevlg{};
FixedStorage<float,(1-1+1),1,1> chcevlg{};
FixedStorage<float,(1-1+1),1,1> vhfzh{};
FixedStorage<float,(1-1+1),1,1> vffzf{};
FixedStorage<float,(1-1+1),1,1> vhlfzhl{};
FixedStorage<float,(1-1+1),1,1> vhshdr{};
FixedStorage<float,(1-1+1),1,1> vhlshdr{};
FixedStorage<float,(1-1+1),1,1> vhmlr{};
FixedStorage<float,(1-1+1),1,1> vhlmlr{};
FixedStorage<float,(1-1+1),1,1> vhsoak{};
FixedStorage<float,(1-1+1),1,1> vhlsoak{};
FixedStorage<float,(1-1+1),1,1> qsdpv{};
FixedStorage<float,(1-1+1),1,1> qssbv{};
FixedStorage<float,(1-1+1),1,1> qsmlr{};
FixedStorage<float,(1-1+1),1,1> qsdsv{};
FixedStorage<float,(1-1+1),1,1> qscev{};
FixedStorage<float,(1-1+1),1,1> qscndv{};
FixedStorage<float,(1-1+1),1,1> qsevv{};
FixedStorage<float,(1-1+1),1,1> qswet{};
FixedStorage<float,(1-1+1),1,1> qsdry{};
FixedStorage<float,(1-1+1),1,1> qsshr{};
FixedStorage<float,(1-1+1),1,1> qsshrp{};
FixedStorage<float,(1-1+1),1,1> qsfzs{};
FixedStorage<float,(1-1+1),1,1> qipdpv{};
FixedStorage<float,(1-1+1),1,1> qipsbv{};
FixedStorage<float,(1-1+1),1,1> qipmlr{};
FixedStorage<float,(1-1+1),1,1> qipdsv{};
FixedStorage<float,(1-1+1),1,1> qirdpv{};
FixedStorage<float,(1-1+1),1,1> qirsbv{};
FixedStorage<float,(1-1+1),1,1> qirmlr{};
FixedStorage<float,(1-1+1),1,1> qirdsv{};
FixedStorage<float,(1-1+1),1,1> qirmlw{};
FixedStorage<float,(1-1+1),1,1> qgldpv{};
FixedStorage<float,(1-1+1),1,1> qglsbv{};
FixedStorage<float,(1-1+1),1,1> qglmlr{};
FixedStorage<float,(1-1+1),1,1> qgldsv{};
FixedStorage<float,(1-1+1),1,1> qglwet{};
FixedStorage<float,(1-1+1),1,1> qgldry{};
FixedStorage<float,(1-1+1),1,1> qglshr{};
FixedStorage<float,(1-1+1),1,1> qglshrp{};
FixedStorage<float,(1-1+1),1,1> qgmdpv{};
FixedStorage<float,(1-1+1),1,1> qgmsbv{};
FixedStorage<float,(1-1+1),1,1> qgmmlr{};
FixedStorage<float,(1-1+1),1,1> qgmdsv{};
FixedStorage<float,(1-1+1),1,1> qgmwet{};
FixedStorage<float,(1-1+1),1,1> qgmdry{};
FixedStorage<float,(1-1+1),1,1> qgmshr{};
FixedStorage<float,(1-1+1),1,1> qgmshrp{};
FixedStorage<float,(1-1+1),1,1> qghdpv{};
FixedStorage<float,(1-1+1),1,1> qghsbv{};
FixedStorage<float,(1-1+1),1,1> qghmlr{};
FixedStorage<float,(1-1+1),1,1> qghdsv{};
FixedStorage<float,(1-1+1),1,1> qghwet{};
FixedStorage<float,(1-1+1),1,1> qghdry{};
FixedStorage<float,(1-1+1),1,1> qghshr{};
FixedStorage<float,(1-1+1),1,1> qghshrp{};
FixedStorage<float,(1-1+1),1,1> qrztot{};
FixedStorage<float,(1-1+1),1,1> qrzmax{};
FixedStorage<float,(1-1+1),1,1> qrzfac{};
FixedStorage<float,(1-1+1),1,1> qrcev{};
FixedStorage<float,(1-1+1),1,1> qrshr{};
FixedStorage<float,(1-1+1),1,1> fsw{};
FixedStorage<float,(1-1+1),1,1> fhw{};
FixedStorage<float,(1-1+1),1,1> fhlw{};
FixedStorage<float,(1-1+1),1,1> ffw{};
FixedStorage<float,(1-1+1),1,1> fswmax{};
FixedStorage<float,(1-1+1),1,1> fhwmax{};
FixedStorage<float,(1-1+1),1,1> fhlwmax{};
FixedStorage<float,(1-1+1),1,1> ffwmax{};
FixedStorage<float,(1-1+1),1,1> qhcnf{};
FixedStorage<float,(1-1+1),1,1> qhlcnh{};
FixedStorage<float,(1-1+1),1,1> qhcngh{};
FixedStorage<float,(1-1+1),1,1> qhcngm{};
FixedStorage<float,(1-1+1),1,1> qhcngl{};
FixedStorage<float,(1-1+1),1,1> qhcnhl{};
FixedStorage<float,(1-1+1),1,1> chcnhl{};
FixedStorage<float,(1-1+1),1,1> zhcnhl{};
FixedStorage<float,(1-1+1),1,1> vhcnhl{};
FixedStorage<float,(1-1+1),1,1> eiw{};
FixedStorage<float,(1-1+1),1,1> eii{};
FixedStorage<float,(1-1+1),1,1> eiri{};
FixedStorage<float,(1-1+1),1,1> eipir{};
FixedStorage<float,(1-1+1),1,1> eisw{};
FixedStorage<float,(1-1+1),1,1> erw{};
FixedStorage<float,(1-1+1),1,1> esw{};
FixedStorage<float,(1-1+1),1,1> eglw{};
FixedStorage<float,(1-1+1),1,1> eghw{};
FixedStorage<float,(1-1+1),1,1> efw{};
FixedStorage<float,(1-1+1),1,1> ehxw{};
FixedStorage<float,(1-1+1),1,1> ehlw{};
FixedStorage<float,(1-1+1),1,1> egmw{};
FixedStorage<float,(1-1+1),1,1> ehw{};
FixedStorage<float,(1-1+1),1,1> err{};
FixedStorage<float,(1-1+1),1,1> esr{};
FixedStorage<float,(1-1+1),1,1> eglr{};
FixedStorage<float,(1-1+1),1,1> eghr{};
FixedStorage<float,(1-1+1),1,1> efr{};
FixedStorage<float,(1-1+1),1,1> ehxr{};
FixedStorage<float,(1-1+1),1,1> ehlr{};
FixedStorage<float,(1-1+1),1,1> egmr{};
FixedStorage<float,(1-1+1),1,1> eri{};
FixedStorage<float,(1-1+1),1,1> esi{};
FixedStorage<float,(1-1+1),1,1> egli{};
FixedStorage<float,(1-1+1),1,1> eghi{};
FixedStorage<float,(1-1+1),1,1> efi{};
FixedStorage<float,(1-1+1),1,1> efis{};
FixedStorage<float,(1-1+1),1,1> ehxi{};
FixedStorage<float,(1-1+1),1,1> ehli{};
FixedStorage<float,(1-1+1),1,1> egmi{};
FixedStorage<float,(1-1+1),1,1> ehi{};
FixedStorage<float,(1-1+1),1,1> ehis{};
FixedStorage<float,(1-1+1),1,1> ehlis{};
FixedStorage<float,(1-1+1),1,1> ers{};
FixedStorage<float,(1-1+1),1,1> ess{};
FixedStorage<float,(1-1+1),1,1> egls{};
FixedStorage<float,(1-1+1),1,1> eghs{};
FixedStorage<float,(1-1+1),1,1> efs{};
FixedStorage<float,(1-1+1),1,1> ehs{};
FixedStorage<float,(1-1+1),1,1> ehsfac{};
FixedStorage<float,(1-1+1),1,1> ehscnv{};
FixedStorage<float,(1-1+1),1,1> ehxs{};
FixedStorage<float,(1-1+1),1,1> ehls{};
FixedStorage<float,(1-1+1),1,1> egms{};
FixedStorage<float,(1-1+1),1,1> egmip{};
FixedStorage<float,(1-1+1),1,1> ehsclsn{};
FixedStorage<float,(1-1+1),1,1> ehiclsn{};
FixedStorage<float,(1-1+1),1,1> ehisclsn{};
FixedStorage<float,(1-1+1),1,1> efsclsn{};
FixedStorage<float,(1-1+1),1,1> eficlsn{};
FixedStorage<float,(1-1+1),1,1> efisclsn{};
FixedStorage<float,(1-1+1),1,1> ehlsclsn{};
FixedStorage<float,(1-1+1),1,1> ehliclsn{};
FixedStorage<float,(1-1+1),1,1> ehlisclsn{};
FixedStorage<float,(1-1+1),1,1> esiclsn{};
float ehs_collsn=0.5f;
float ehi_collsn=1.0f;
float efs_collsn=0.5f;
float efi_collsn=1.0f;
float ehls_collsn=1.0f;
float ehli_collsn=1.0f;
float esi_collsn=1.0f;
FixedStorage<float,(8-1+1)*(6-1+1),1,8,1,6> ew{};
FixedStorage<float,(8-1+1)*(2-1+1),1,8,1,2> cwr{};
FixedStorage<int,(1-1+1),1,1> icwr{};
FixedStorage<int,(1-1+1),1,1> igwr{};
FixedStorage<int,(1-1+1),1,1> irwr{};
FixedStorage<int,(1-1+1),1,1> ihlr{};
FixedStorage<int,(1-1+1),1,1> ifwr{};
FixedStorage<float,(6-1+1)*(2-1+1),1,6,1,2> grad{};
FixedStorage<float,(1-1+1),1,1> da0lr{};
FixedStorage<float,(1-1+1),1,1> da1lr{};
FixedStorage<float,(1-1+1),1,1> da0lc{};
FixedStorage<float,(1-1+1),1,1> da1lc{};
FixedStorage<float,(1-1+1),1,1> da0lh{};
FixedStorage<float,(1-1+1),1,1> da0lhl{};
FixedStorage<float,(1-1+1),1,1> da0lf{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> da0lx{};
FixedStorage<float,(lqmx-lc+1),lc,lqmx> va0{};
FixedStorage<float,(lqmx-lc+1)*(lqmx-lc+1),lc,lqmx,lc,lqmx> vab0{};
FixedStorage<float,(lqmx-lc+1)*(lqmx-lc+1),lc,lqmx,lc,lqmx> vab1{};
FixedStorage<float,(lqmx-lc+1),lc,lqmx> va1{};
FixedStorage<float,(1-1+1),1,1> ehip{};
FixedStorage<float,(1-1+1),1,1> ehlip{};
FixedStorage<float,(1-1+1),1,1> ehlir{};
FixedStorage<float,(1-1+1),1,1> erir{};
FixedStorage<float,(1-1+1),1,1> esir{};
FixedStorage<float,(1-1+1),1,1> eglir{};
FixedStorage<float,(1-1+1),1,1> egmir{};
FixedStorage<float,(1-1+1),1,1> eghir{};
FixedStorage<float,(1-1+1),1,1> efir{};
FixedStorage<float,(1-1+1),1,1> ehir{};
FixedStorage<float,(1-1+1),1,1> eirw{};
FixedStorage<float,(1-1+1),1,1> eirir{};
FixedStorage<float,(1-1+1),1,1> ehr{};
FixedStorage<float,(1-1+1),1,1> erip{};
FixedStorage<float,(1-1+1),1,1> esip{};
FixedStorage<float,(1-1+1),1,1> eglip{};
FixedStorage<float,(1-1+1),1,1> eghip{};
FixedStorage<float,(1-1+1),1,1> efip{};
FixedStorage<float,(1-1+1),1,1> eipi{};
FixedStorage<float,(1-1+1),1,1> eipw{};
FixedStorage<float,(1-1+1),1,1> eipip{};
FixedStorage<float,(1-1+1),1,1> ptotal{};
FixedStorage<float,(1-1+1),1,1> pqcwi{};
FixedStorage<float,(1-1+1),1,1> pqcii{};
FixedStorage<float,(1-1+1),1,1> pqrwi{};
FixedStorage<float,(1-1+1),1,1> pqisi{};
FixedStorage<float,(1-1+1),1,1> pqswi{};
FixedStorage<float,(1-1+1),1,1> pqhwi{};
FixedStorage<float,(1-1+1),1,1> pqwvi{};
FixedStorage<float,(1-1+1),1,1> pqgli{};
FixedStorage<float,(1-1+1),1,1> pqghi{};
FixedStorage<float,(1-1+1),1,1> pqfwi{};
FixedStorage<float,(1-1+1),1,1> pqgmi{};
FixedStorage<float,(1-1+1),1,1> pqhli{};
FixedStorage<float,(1-1+1),1,1> pqiri{};
FixedStorage<float,(1-1+1),1,1> pqipi{};
FixedStorage<float,(1-1+1),1,1> pqlwsi{};
FixedStorage<float,(1-1+1),1,1> pqlwhi{};
FixedStorage<float,(1-1+1),1,1> pqlwhli{};
FixedStorage<float,(1-1+1),1,1> pqlwfi{};
FixedStorage<float,(1-1+1),1,1> pqlwlghi{};
FixedStorage<float,(1-1+1),1,1> pqlwlghli{};
FixedStorage<float,(1-1+1),1,1> pqlwlghd{};
FixedStorage<float,(1-1+1),1,1> pqlwlghld{};
FixedStorage<float,(1-1+1),1,1> pvhwi{};
FixedStorage<float,(1-1+1),1,1> pvhwd{};
FixedStorage<float,(1-1+1),1,1> pvfwi{};
FixedStorage<float,(1-1+1),1,1> pvfwd{};
FixedStorage<float,(1-1+1),1,1> pvhli{};
FixedStorage<float,(1-1+1),1,1> pvhld{};
FixedStorage<float,(1-1+1),1,1> pvswi{};
FixedStorage<float,(1-1+1),1,1> pvswd{};
FixedStorage<float,(1-1+1),1,1> pqcwd{};
FixedStorage<float,(1-1+1),1,1> pqcid{};
FixedStorage<float,(1-1+1),1,1> pqrwd{};
FixedStorage<float,(1-1+1),1,1> pqisd{};
FixedStorage<float,(1-1+1),1,1> pqcwdacc{};
FixedStorage<float,(1-1+1),1,1> pqswd{};
FixedStorage<float,(1-1+1),1,1> pqhwd{};
FixedStorage<float,(1-1+1),1,1> pqwvd{};
FixedStorage<float,(1-1+1),1,1> pqgld{};
FixedStorage<float,(1-1+1),1,1> pqghd{};
FixedStorage<float,(1-1+1),1,1> pqfwd{};
FixedStorage<float,(1-1+1),1,1> pqgmd{};
FixedStorage<float,(1-1+1),1,1> pqhld{};
FixedStorage<float,(1-1+1),1,1> pqird{};
FixedStorage<float,(1-1+1),1,1> pqipd{};
FixedStorage<float,(1-1+1),1,1> pqlwsd{};
FixedStorage<float,(1-1+1),1,1> pqlwhd{};
FixedStorage<float,(1-1+1),1,1> pqlwhld{};
FixedStorage<float,(1-1+1),1,1> pqlwfd{};
FixedStorage<float,(1-1+1),1,1> pctot{};
FixedStorage<float,(1-1+1),1,1> pcipi{};
FixedStorage<float,(1-1+1),1,1> pcipd{};
FixedStorage<float,(1-1+1),1,1> pciri{};
FixedStorage<float,(1-1+1),1,1> pcird{};
FixedStorage<float,(1-1+1),1,1> pccwi{};
FixedStorage<float,(1-1+1),1,1> pccwd{};
FixedStorage<float,(1-1+1),1,1> pccwdacc{};
FixedStorage<float,(1-1+1),1,1> pccii{};
FixedStorage<float,(1-1+1),1,1> pccid{};
FixedStorage<float,(1-1+1),1,1> pcisi{};
FixedStorage<float,(1-1+1),1,1> pcisd{};
FixedStorage<float,(1-1+1),1,1> pccin{};
FixedStorage<float,(1-1+1),1,1> pcrwi{};
FixedStorage<float,(1-1+1),1,1> pcrwd{};
FixedStorage<float,(1-1+1),1,1> pcswi{};
FixedStorage<float,(1-1+1),1,1> pcswd{};
FixedStorage<float,(1-1+1),1,1> pchwi{};
FixedStorage<float,(1-1+1),1,1> pchwd{};
FixedStorage<float,(1-1+1),1,1> pchli{};
FixedStorage<float,(1-1+1),1,1> pchld{};
FixedStorage<float,(1-1+1),1,1> pcfwi{};
FixedStorage<float,(1-1+1),1,1> pcfwd{};
FixedStorage<float,(1-1+1),1,1> pcgli{};
FixedStorage<float,(1-1+1),1,1> pcgld{};
FixedStorage<float,(1-1+1),1,1> pcgmi{};
FixedStorage<float,(1-1+1),1,1> pcgmd{};
FixedStorage<float,(1-1+1),1,1> pcghi{};
FixedStorage<float,(1-1+1),1,1> pcghd{};
FixedStorage<float,(1-1+1),1,1> pzrwi{};
FixedStorage<float,(1-1+1),1,1> pzrwd{};
FixedStorage<float,(1-1+1),1,1> pzhwi{};
FixedStorage<float,(1-1+1),1,1> pzhwd{};
FixedStorage<float,(1-1+1),1,1> pzfwi{};
FixedStorage<float,(1-1+1),1,1> pzfwd{};
FixedStorage<float,(1-1+1),1,1> pzhli{};
FixedStorage<float,(1-1+1),1,1> pzhld{};
FixedStorage<float,(1-1+1),1,1> pzswi{};
FixedStorage<float,(1-1+1),1,1> pzswd{};
FixedStorage<float,(1-1+1),1,1> dqisdt{};
FixedStorage<float,(1-1+1),1,1> qss0{};
FixedStorage<float,(1-1+1),1,1> qsacip{};
FixedStorage<float,(1-1+1),1,1> pres{};
FixedStorage<float,(1-1+1),1,1> pipert{};
FixedStorage<float,(1-1+1),1,1> pk{};
FixedStorage<float,(1-1+1),1,1> rho0{};
FixedStorage<float,(1-1+1),1,1> pi0{};
FixedStorage<float,(1-1+1),1,1> rhovt{};
float sqrtrhovt;
FixedStorage<float,(1-1+1),1,1> thetap{};
FixedStorage<float,(1-1+1),1,1> theta0{};
FixedStorage<float,(1-1+1),1,1> qwvp{};
FixedStorage<float,(1-1+1),1,1> qv0{};
FixedStorage<float,(1-1+1),1,1> thsave{};
FixedStorage<float,(1-1+1),1,1> ptwfzi{};
FixedStorage<float,(1-1+1),1,1> ptimlw{};
FixedStorage<float,(1-1+1),1,1> psub{};
FixedStorage<float,(1-1+1),1,1> pvap{};
FixedStorage<float,(1-1+1),1,1> pfrz{};
FixedStorage<float,(1-1+1),1,1> ptem{};
FixedStorage<float,(1-1+1),1,1> pmlt{};
FixedStorage<float,(1-1+1),1,1> pevap{};
FixedStorage<float,(1-1+1),1,1> pdep{};
FixedStorage<float,(1-1+1),1,1> ptem2{};
FixedStorage<float,(1-1+1),1,1> cnostmp{};
constexpr int iholef=1;
constexpr int iholen=1;
float cqtotn;
float cqtotn1;
float cctotn;
float citotn;
float crtotn;
float cstotn;
float cvtotn;
float cftotn;
float cgltotn;
float cghtotn;
float chtotn;
float cqtotp;
float cqtotp1;
float cctotp;
float citotp;
float ciptotp;
float crtotp;
float cstotp;
float cvtotp;
float cftotp;
float chltotp;
float cgltotp;
float cgmtotp;
float cghtotp;
float chtotp;
float cqfac;
float ccfac;
float cifac;
float cipfac;
float crfac;
float csfac;
float cvfac;
float cffac;
float cglfac;
float cghfac;
float chfac;
float ssifac;
float qvapor;
const float cwmas30=0x1.f168540000000p-34f;
const float cwmas20=0x1.26c2900000000p-35f;
int ireadqf;
int lrho;
int lqsw;
int lqgl;
int lqgm;
int lqgh;
int lqrw;
float vt;
float arg;
float erbnd1;
float fdgt1;
float costhe1;
float qeps;
float dyi2;
float dzi2;
float bta1;
float cnit;
float dragh;
float dnz00;
float pii;
float qccrit;
float gf4br;
float gf4ds;
float gf4p5;
float gf3ds;
float gf1ds;
FixedStorage<float,(1-1+1),1,1> gf1palp{};
float xdn_new;
float drhodt;
int l;
int ltemq;
int inumgs;
int idelq;
float brz;
float arz;
float temq;
float ssival;
float tqvcon;
float cnox;
float cval;
float aval;
float eval;
float fval;
float gval;
float qsign;
float ftelwc;
float qconkq;
float elecfac;
float altelecfac;
float qconm;
float qconn;
float cfce15;
float gf8;
float gf4i;
float gf3p5;
float gf1a;
float gf1p5;
float qdiff;
float argrcnw;
float c4;
float bradp;
float bl2;
float bt2;
float dthr;
float hrifac;
float hdia0;
float hdia1;
float civenta;
float civentb;
float civentc;
float civentd;
float civente;
float civentf;
float civentg;
float cireyn;
float xcivent;
float cipventa;
float cipventb;
float cipventc;
float cipventd;
float cipreyn;
float cirventa;
float cirventb;
int igmrwa;
int igmrwb;
int igmswa;
int igmswb;
int igmfwa;
int igmfwb;
int igmhwa;
int igmhwb;
float rwventa;
float rwventb;
float swventa;
float swventb;
float fwventa;
float fwventb;
float fwventc;
float hwventa;
float hwventb;
float hwventc;
float hlventa;
float hlventb;
float hlventc;
float glventa;
float glventb;
float glventc;
float gmventa;
float gmventb;
float gmventc;
float ghventa;
float ghventb;
float ghventc;
float dzfacp;
float dzfacm;
float cmassin;
float cwdiar;
float rimmas;
float rhobar;
float argtim;
float argqcw;
float argqxw;
float argtem;
float frcswsw;
float frcswgl;
float frcswgm;
float frcswgh;
float frcswfw;
float frcswsw1;
float frcglgl;
float frcglgm;
float frcglgh;
float frcglfw;
float frcglgl1;
float frcgmgl;
float frcgmgm;
float frcgmgh;
float frcgmfw;
float frcgmgm1;
float frcghgl;
float frcghgm;
float frcghgh;
float frcghfw;
float frcghgh1;
float frcfwgl;
float frcfwgm;
float frcfwgh;
float frcfwfw;
float frcfwfw1;
float frcswrsw;
float frcswrgl;
float frcswrgm;
float frcswrgh;
float frcswrfw;
float frcswrsw1;
float frcrswsw;
float frcrswgl;
float frcrswgm;
float frcrswgh;
float frcrswfw;
float frcrswsw1;
float frcglrgl;
float frcglrgm;
float frcglrgh;
float frcglrfw;
float frcglrgl1;
float frcrglgl;
float frcrglgm;
float frcrglgh;
float frcrglfw;
float frcrglgl1;
float frcgmrgl;
float frcgmrgm;
float frcgmrgh;
float frcgmrfw;
float frcgmrgm1;
float frcrgmgl;
float frcrgmgm;
float frcrgmgh;
float frcrgmfw;
float frcrgmgm1;
float total;
float qweps;
float gf2a;
float gf4a;
float dqldt;
float dqidt;
float dqdt;
float frcghrgl;
float frcghrgm;
float frcghrgh;
float frcghrfw;
float frcghrgh1;
float frcrghgl;
float frcrghgm;
float frcrghgh;
float frcrghfw;
float frcrghgh1;
float a1;
float a2;
float a3;
float a4;
float a5;
float a6;
float gamss;
float cdw;
float cdi;
float denom1;
float denom2;
float delqci1;
float delqip1;
float cirtotn;
float ciptotn;
float cgmtotn;
float chltotn;
float cirtotp;
float cgmfac;
float chlfac;
float cirfac;
int igmhla;
int igmhlb;
int igmgla;
int igmglb;
int igmgma;
int igmgmb;
int igmgha;
int igmghb;
int idqis;
int item;
int itim0;
int iqgl;
int iqgm;
int iqgh;
int iqrw;
int iqsw;
int itertd;
int ia;
int infdo;
float tau;
float ewtmp;
int cntnic_noliq;
float q_noliqmn;
float q_noliqmx;
float scsacimn;
float scsacimx;
float dtpinv;
float xden;
float xmlt;
float cmlt;
float cmlttot;
float fventm;
float fventh;
float am;
float ah;
float felfinv;
float dmwdt;
float qhmlrtmp;
float qhmlrtmp2;
float chmlrtmp;
float chmlrtmpd1inf;
float chlmlrtmp;
float zhlmlrtmp;
float zhlmlrrtmp;
float qvs0;
float tmpcmlt;
float term1;
float term2;
float term3;
float term4;
float qaacw;
float cwchtmp;
const float c1r=0x1.3000000000000p+4f;
const float c2r=0x1.3333340000000p-1f;
const float c3r=0x1.cccccc0000000p+0f;
const float c4r=0x1.1000000000000p+4f;
const float c1h=0x1.6000000000000p+2f;
const float c2h=0x1.6666660000000p-1f;
const float c3h=0x1.2000000000000p+2f;
const float c4h=0x1.1000000000000p+3f;
const float c1hl=0x1.d9999a0000000p+1f;
const float c2hl=0x1.3333340000000p-2f;
const float c3hl=0x1.2000000000000p+3f;
const float c4hl=0x1.a000000000000p+2f;
const float c5hl=0x1.0000000000000p+0f;
const float c6hl=0x1.a000000000000p+2f;
float a_in;
constexpr bool newton=false;
// WRF:13066
cwr.data[0]=2.0f;cwr.data[1]=3.0f;cwr.data[2]=4.0f;cwr.data[3]=6.0f;cwr.data[4]=8.0f;cwr.data[5]=10.0f;cwr.data[6]=15.0f;cwr.data[7]=20.0f;cwr.data[8]=1.0f;cwr.data[9]=1.0f;cwr.data[10]=0.5f;cwr.data[11]=0.5f;cwr.data[12]=0.5f;cwr.data[13]=0.2f;cwr.data[14]=0.2f;cwr.data[15]=1.f;
// WRF:13070
grad.data[0]=100.f;grad.data[1]=200.f;grad.data[2]=300.f;grad.data[3]=400.f;grad.data[4]=600.f;grad.data[5]=1000.f;grad.data[6]=1.e-2f;grad.data[7]=1.e-2f;grad.data[8]=1.e-2f;grad.data[9]=5.e-3f;grad.data[10]=2.5e-3f;grad.data[11]=1.f;
// WRF:13073
ew.data[0]=0.03f;ew.data[1]=0.07f;ew.data[2]=0.17f;ew.data[3]=0.41f;ew.data[4]=0.58f;ew.data[5]=0.69f;ew.data[6]=0.82f;ew.data[7]=0.88f;ew.data[8]=0.10f;ew.data[9]=0.20f;ew.data[10]=0.34f;ew.data[11]=0.58f;ew.data[12]=0.70f;ew.data[13]=0.78f;ew.data[14]=0.88f;ew.data[15]=0.92f;ew.data[16]=0.15f;ew.data[17]=0.31f;ew.data[18]=0.44f;ew.data[19]=0.65f;ew.data[20]=0.75f;ew.data[21]=0.83f;ew.data[22]=0.96f;ew.data[23]=0.91f;ew.data[24]=0.17f;ew.data[25]=0.37f;ew.data[26]=0.50f;ew.data[27]=0.70f;ew.data[28]=0.81f;ew.data[29]=0.87f;ew.data[30]=0.93f;ew.data[31]=0.96f;ew.data[32]=0.17f;ew.data[33]=0.40f;ew.data[34]=0.54f;ew.data[35]=0.71f;ew.data[36]=0.83f;ew.data[37]=0.88f;ew.data[38]=0.94f;ew.data[39]=0.98f;ew.data[40]=0.15f;ew.data[41]=0.37f;ew.data[42]=0.52f;ew.data[43]=0.74f;ew.data[44]=0.82f;ew.data[45]=0.88f;ew.data[46]=0.94f;ew.data[47]=0.98f;
// WRF:13320
auto galpha=[&](float a_in) {return lake_div((((((4.f+a_in))*((5.f+a_in)))*((6.f+a_in)))),(((((1.f+a_in))*((2.f+a_in)))*((3.f+a_in)))));};
// WRF:13321
auto dgalpha=[&](float a_in) {return lake_div((((((876.f+(1260.f*a_in))+(621.f*lake_pow(a_in,2)))+(126.f*lake_pow(a_in,3)))+(9.f*lake_pow(a_in,4)))),(((((((36.f+(132.f*a_in))+(193.f*lake_pow(a_in,2)))+(144.f*lake_pow(a_in,3)))+(58.f*lake_pow(a_in,4)))+(12.f*lake_pow(a_in,5)))+lake_pow(a_in,6))));};
// WRF:13334
for(int _s0=((-norz)+ng1);_s0<=(nz+norz);++_s0) {pb(_s0) = 0.0f;}
// WRF:13335
for(int _s0=((-norz)+ng1);_s0<=(nz+norz);++_s0) {pinit(_s0) = 0.0f;}
// WRF:13336
itile = nx;
// WRF:13337
jtile = ny;
// WRF:13338
ktile = nz;
// WRF:13339
ixend = nx;
// WRF:13340
jyend = ny;
// WRF:13341
kzend = nz;
// WRF:13342
nxend = (nx+1);
// WRF:13343
nyend = (ny+1);
// WRF:13344
nzend = nz;
// WRF:13345
kzbeg = 1;
// WRF:13346
nzbeg = 1;
// WRF:13348
istag = 0;
// WRF:13349
jstag = 0;
// WRF:13350
kstag = 1;
// WRF:13352
for(int _s0=lc;_s0<=lhab;++_s0) {lrescalelow(_s0) = rescale_low_alpha;}
// WRF:13353
lrescalelow(lr) = (rescale_low_alphar&&rescale_low_alpha);
// WRF:13354
lrescalelow(lh) = (rescale_low_alphah&&rescale_low_alpha);
// WRF:13355
if((lf>1)) {lrescalelow(lf) = (rescale_low_alphah&&rescale_low_alpha);}
// WRF:13356
if((lhl>1)) {lrescalelow(lhl) = (rescale_low_alphahl&&rescale_low_alpha);}
// WRF:13363
if((ngs<nz)) {
// WRF:13366
}
// WRF:13368
cntnic_noliq = 0;
// WRF:13369
q_noliqmn = 0.0f;
// WRF:13370
q_noliqmx = 0.0f;
// WRF:13371
scsacimn = 0.0f;
// WRF:13372
scsacimx = 0.0f;
// WRF:13374
ldovol = false;
// WRF:13376
#pragma unroll
for(il=lc;il<=lhab;il+=1) {
// WRF:13377
ldovol = (ldovol||((coeff_lvol{}(il)>1)));
// WRF:13378
}
// WRF:13381
ffrzh = 1;
// WRF:13395
dtpinv = lake_div(1.e0,dtp);
// WRF:13404
qeps = 1.0e-20f;
// WRF:13414
aradcw = (-0.27544f);
// WRF:13415
bradcw = 0.26249e+06f;
// WRF:13416
cradcw = (-1.8896e+10f);
// WRF:13417
dradcw = 4.4626e+14f;
// WRF:13418
bta1 = 0.6f;
// WRF:13419
cnit = 1.0e-02f;
// WRF:13420
dragh = 0.60f;
// WRF:13421
dnz00 = 1.225f;
// WRF:13425
cs = 12.42f;
// WRF:13426
ds = 0.42f;
// WRF:13427
pii = piinv;
// WRF:13428
pid4 = lake_div(pi,4.0f);
// WRF:13430
gf1 = 1.0f;
// WRF:13431
gf1p5 = 0.8862269255f;
// WRF:13432
gf2 = 1.0f;
// WRF:13433
gf3 = 2.0f;
// WRF:13434
gf3p5 = 3.32335097f;
// WRF:13435
gf4 = 6.00f;
// WRF:13436
gf5 = 24.0f;
// WRF:13437
gf6 = 120.0f;
// WRF:13438
gf7 = 720.0f;
// WRF:13439
gf4br = 17.837861981813607f;
// WRF:13440
gf4ds = 10.41688578110938f;
// WRF:13441
gf4p5 = 11.63172839656745f;
// WRF:13442
gf3ds = 3.0458730354120997f;
// WRF:13443
gf1ds = 0.8863557896089221f;
// WRF:13445
gf43rds = 0.8929795116f;
// WRF:13446
gf53rds = 0.9027452930f;
// WRF:13447
gf73rds = 1.190639349f;
// WRF:13448
gf83rds = 1.504575488f;
// WRF:13450
gamice73fac = lake_div(lake_pow((gamma_sp((lake_div(7.f,3.f)+cinu))),3),((lake_pow(gamma_sp((1.f+cinu)),3)*lake_pow(((1.f+cinu)),4))));
// WRF:13451
gamsnow73fac = lake_div(lake_pow((gamma_sp((lake_div(7.f,3.f)+snu))),3),((lake_pow(gamma_sp((1.f+snu)),3)*lake_pow(((1.f+snu)),4))));
// WRF:13461
brz = 100.0f;
// WRF:13462
arz = 0.66f;
// WRF:13464
bfnu1 = lake_div(((((4.f+alphar))*((5.f+alphar)))*((6.f+alphar))),(((((1.f+alphar))*((2.f+alphar)))*((3.f+alphar)))));
// WRF:13467
galpharaut = lake_div(((((6.f+alpharaut))*((5.f+alpharaut)))*((4.f+alpharaut))),(((((3.f+alpharaut))*((2.f+alpharaut)))*((1.f+alpharaut)))));
// WRF:13470
vfrz = (0.523599f*lake_pow((dfrz),3));
// WRF:13471
vmlt = lake_min(coeff_xvmx{}(lr),(0.523599f*lake_pow((dmlt),3)));
// WRF:13472
vshd = lake_min(coeff_xvmx{}(lr),(0.523599f*lake_pow((dshd),3)));
// WRF:13474
if((snowmeltdia>0.0f)) {
// WRF:13475
snowmeltmass = ((lake_div(pi,6.0f)*1000.f)*lake_pow(snowmeltdia,3));
// WRF:13476
}
// WRF:13478
tdtol = 1.0e-05f;
// WRF:13479
tfrcbw = (tfr-cbw);
// WRF:13480
tfrcbi = (tfr-cbi);
// WRF:13482
if(mixedphase) {
// WRF:13483
ibinhmlr = 0;
// WRF:13484
ibinhlmlr = 0;
// WRF:13485
}
// WRF:13501
mltmass0inv = lake_div(1.0f,((1000.0f*coeff_xvmx{}(lr))));
// WRF:13502
mltmass1inv = lake_div(1.0f,(((1000.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow((((0.01f*0.5f)*takshedsize1)),3)))));
// WRF:13503
mltmass2inv = lake_div(1.0f,(((1000.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow((((0.01f*0.5f)*takshedsize2)),3)))));
// WRF:13504
mltmass3inv = lake_div(1.0f,(((1000.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow((((0.01f*0.5f)*takshedsize3)),3)))));
// WRF:13505
mltmass1cgs = ((1.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow(((0.5f*takshedsize1)),3)));
// WRF:13506
mltmass2cgs = ((1.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow(((0.5f*takshedsize2)),3)));
// WRF:13507
mltmass3cgs = ((1.0f*(lake_div((4.0f*pi),3.0f)))*(lake_pow(((0.5f*takshedsize3)),3)));
// WRF:13511
if((ibinnum==1)) {
// WRF:13512
numdiam = 1;
// WRF:13513
mltdiam(1) = 4.5e-3f;
// WRF:13514
} else if((ibinnum==2)) {
// WRF:13515
numdiam = 2;
// WRF:13516
mltdiam(1) = lake_div(mltdiam1,6.f);
// WRF:13517
mltdiam(2) = lake_div(mltdiam1,2.f);
// WRF:13518
} else if((ibinnum>2)) {
// WRF:13519
numdiam = lake_min(ibinnum,ndiam);
// WRF:13520
for(k=1;k<=numdiam;k+=1) {
// WRF:13521
mltdiam(k) = lake_div((((k-0.5f))*mltdiam1),float(numdiam));
// WRF:13522
}
// WRF:13524
} else {
// WRF:13525
numdiam = 5;
// WRF:13526
mltdiam(1) = 0.5e-3f;
// WRF:13527
mltdiam(2) = 1.0e-3f;
// WRF:13528
mltdiam(3) = 2.0e-3f;
// WRF:13529
mltdiam(4) = 4.0e-3f;
// WRF:13530
mltdiam(5) = 6.0e-3f;
// WRF:13531
}
// WRF:13534
if((numshedregimes==2)) {
// WRF:13535
mltdiam((ndiam+1)) = mltdiam1;
// WRF:13536
mltdiam((ndiam+2)) = mltdiam3;
// WRF:13537
mltdiam((ndiam+3)) = mltdiam4;
// WRF:13538
} else if((numshedregimes==3)) {
// WRF:13539
mltdiam((ndiam+1)) = mltdiam1;
// WRF:13540
mltdiam((ndiam+2)) = mltdiam2;
// WRF:13541
mltdiam((ndiam+3)) = mltdiam3;
// WRF:13542
mltdiam((ndiam+4)) = mltdiam4;
// WRF:13543
}
// WRF:13545
kzb = 1;
// WRF:13546
kze = ktile;
// WRF:13553
mwfac = lake_pow(6.0f,(lake_div(1.f,3.f)));
// WRF:13554
if((ipconc>=2)) {
// WRF:13558
}
// WRF:13559
rwmasn = (coeff_xvmn{}(lr)*1000.f);
// WRF:13560
rwmasx = (coeff_xvmx{}(lr)*1000.f);
// WRF:13562
if((biggsnowdiam>0.0f)) {
// WRF:13563
xvbiggsnow = ((lake_div(pi,6.0f))*lake_pow(biggsnowdiam,3));
// WRF:13564
} else {
// WRF:13565
xvbiggsnow = coeff_xvmn{}(lh);
// WRF:13566
}
// WRF:13571
cimasn = lake_min(cimas0,cimas1);
// WRF:13572
cimasx = 1.0e-8f;
// WRF:13573
ccimx = 5000.0e3f;
// WRF:13582
iend = 0;
// WRF:13598
jy = jgs;
// WRF:13609
if((ipconc<2)) {
// WRF:13610
for(kz=1;kz<=kze;kz+=1) {
// WRF:13611
for(ix=1;ix<=itile;ix+=1) {
// WRF:13612
t9(ix,jy,kz) = an(ix,jy,kz,lc);
// WRF:13613
}
// WRF:13614
}
// WRF:13615
}
// WRF:13620
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:13624
nxmpb = 1;
// WRF:13625
nzmpb = 1;
// WRF:13626
nxz = (itile*nz);
// WRF:13627
numgs = (lake_div(nxz,ngs)+1);
{ kz=ix=1;
// WRF:13636
pqs(1) = t00(ix,jy,kz);
// WRF:13638
theta(1) = an(ix,jy,kz,lt);
// WRF:13639
temg(1) = t0(ix,jy,kz);
// WRF:13640
temcg(1) = (temg(1)-tfr);
// WRF:13641
tqvcon = (temg(1)-cbw);
// WRF:13642
ltemq = (lake_div(((temg(1)-163.15f)),fqsat)+1.5f);
// WRF:13643
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:13644
qvs(1) = (pqs(1)*tabqvs(ltemq));
// WRF:13645
if(((iqis0==1)||(temg(1)<=(tfr+0.5f)))) {
// WRF:13646
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:13647
} else {
// WRF:13648
ltemq = (lake_div(((tfr-163.15f)),fqsat)+1.5f);
// WRF:13649
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:13650
}
// WRF:13652
qss(1) = qvs(1);
// WRF:13654
if((temg(1)<tfr)) {
// WRF:13655
qss(1) = qis(1);
// WRF:13656
}
// WRF:13658
ishail = false;
// WRF:13659
if((lhl>1)) {
// WRF:13660
if((an(ix,jy,kz,lhl)>coeff_qxmin{}(lhl))) {ishail = true;}
// WRF:13661
}
// WRF:13665
if((((((((an(ix,jy,kz,lv)>qss(1))||(an(ix,jy,kz,lc)>coeff_qxmin{}(lc)))||(an(ix,jy,kz,li)>coeff_qxmin{}(li)))||(an(ix,jy,kz,lr)>coeff_qxmin{}(lr)))||(an(ix,jy,kz,ls)>coeff_qxmin{}(ls)))||(an(ix,jy,kz,lh)>coeff_qxmin{}(lh)))||ishail)) {
ngscnt=1; } else { ngscnt=0; }
// WRF:13681
if((ngscnt==0)) {goto L9998;}
// WRF:13683
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:13688
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {xv(_s0,_s1) = 0.0f;}}
// WRF:13689
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {xmas(_s0,_s1) = 0.0f;}}
// WRF:13690
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {for(int _s2=1;_s2<=3;++_s2) {vtxbar(_s0,_s1,_s2) = 0.0f;}}}
// WRF:13691
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {for(int _s2=1;_s2<=3;++_s2) {xdia(_s0,_s1,_s2) = 0.0f;}}}
// WRF:13692
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=li;_s1<=lhab;++_s1) {raindn(_s0,_s1) = 900.f;}}
// WRF:13693
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {cx(_s0,_s1) = 0.0f;}}
// WRF:13694
if(((lnhf>1)||(lnhlf>1))) {for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lh;_s1<=lhab;++_s1) {chxf(_s0,_s1) = 0.0f;}}}
// WRF:13695
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {alpha(_s0,_s1) = 0.0f;}}
// WRF:13696
#pragma unroll
for(il=li;il<=lhab;il+=1) {
// WRF:13697
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13698
rimdn(1,il) = rimedens;
// WRF:13699
}
// WRF:13700
}
// WRF:13704
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:13705
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13706
kgsm(1) = lake_max((1-1),1);
// WRF:13707
kgsp(1) = lake_min((1+1),(nz-1));
// WRF:13708
kgsm2(1) = lake_max((1-2),1);
// WRF:13709
theta0(1) = an(1,jy,1,lt);
// WRF:13710
thetap(1) = (an(1,jy,1,lt)-theta0(1));
// WRF:13711
theta(1) = an(1,jy,1,lt);
// WRF:13712
qv0(1) = an(1,jy,1,lv);
// WRF:13713
qwvp(1) = (an(1,jy,1,lv)-qv0(1));
// WRF:13715
pres(1) = (pn(1,jy,1)+pb(1));
// WRF:13716
pipert(1) = p2(1,jy,1);
// WRF:13717
rho0(1) = dn(1,jy,1);
// WRF:13718
rhoinv(1) = lake_div(1.0f,rho0(1));
// WRF:13719
rhovt(1) = sqrtf(lake_div(rho00,lake_max(0.05f,rho0(1))));
// WRF:13720
pi0(1) = (p2(1,jy,1)+pinit(1));
// WRF:13721
temg(1) = t0(1,jy,1);
// WRF:13722
temgkm1(1) = below_t;
// WRF:13723
temgkm2(1) = below2_t;
// WRF:13724
pk(1) = (p2(1,jy,1)+pinit(1));
// WRF:13725
temcg(1) = (temg(1)-tfr);
// WRF:13726
qss0(1) = lake_div((380.0f),(pres(1)));
// WRF:13727
pqs(1) = lake_div((380.0f),(pres(1)));
// WRF:13728
ltemq = (lake_div(((temg(1)-163.15f)),fqsat)+1.5f);
// WRF:13729
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:13730
qvs(1) = (pqs(1)*tabqvs(ltemq));
// WRF:13731
if(((iqis0==1)||(temg(1)<=(tfr+0.5f)))) {
// WRF:13732
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:13733
} else {
// WRF:13734
ltemq = (lake_div(((tfr-163.15f)),fqsat)+1.5f);
// WRF:13735
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:13736
}
// WRF:13737
qss(1) = qvs(1);
// WRF:13740
cnostmp(1) = coeff_cno{}(ls);
// WRF:13743
il5(1) = 0;
// WRF:13744
if((temg(1)<tfr)) {
// WRF:13745
il5(1) = 1;
// WRF:13746
}
// WRF:13747
}
// WRF:13749
if(((ipconc<1)&&lwsm6)) {
// WRF:13750
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13751
tmp = lake_min(0.0f,temcg(1));
// WRF:13752
cnostmp(1) = lake_min(2.e8f,(2.e6f*gfk_exp((0.12f*tmp))));
// WRF:13753
}
// WRF:13754
}
// WRF:13760
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13761
qhshr(1) = 0.0f;
// WRF:13762
}
// WRF:13766
#pragma unroll
for(il=lv;il<=lhab;il+=1) {
// WRF:13767
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13768
qx(1,il) = lake_max(an(1,jy,1,il),0.0f);
// WRF:13769
}
// WRF:13770
}
// WRF:13772
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=ls;_s1<=lhab;++_s1) {qxw(_s0,_s1) = 0.0f;}}
// WRF:13773
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lh;_s1<=lhab;++_s1) {qxwlg(_s0,_s1) = 0.0f;}}
// WRF:13784
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:13786
if((ipconc>=1)) {
// WRF:13787
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13788
cx(1,li) = lake_max(an(1,jy,1,lni),0.0f);
// WRF:13789
if((qx(1,li)<=coeff_qxmin{}(li))) {
// WRF:13790
cx(1,li) = 0.0f;
// WRF:13791
}
// WRF:13793
if((lcina>1)) {
// WRF:13794
cina(1) = an(1,jy,1,lcina);
// WRF:13795
} else {
// WRF:13796
cina(1) = cx(1,li);
// WRF:13797
}
// WRF:13798
if((lcin>1)) {
// WRF:13799
ccin(1) = an(1,jy,1,lcin);
// WRF:13800
}
// WRF:13801
}
// WRF:13802
}
// WRF:13803
if((ipconc>=2)) {
// WRF:13804
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13805
cx(1,lc) = lake_max(an(1,jy,1,lnc),0.0f);
// WRF:13807
if((qx(1,lc)<=coeff_qxmin{}(lc))) {
// WRF:13808
cx(1,lc) = 0.0f;
// WRF:13809
}
// WRF:13810
if((lss>1)) {
// WRF:13811
ssmax(1) = an(1,jy,1,lss);
// WRF:13812
}
// WRF:13813
if((lccn>1)) {
// WRF:13814
ccnc(1) = an(1,jy,1,lccn);
// WRF:13815
} else {
// WRF:13816
ccnc(1) = 0.0f;
// WRF:13817
}
// WRF:13818
if((lccna>1)) {
// WRF:13819
ccna(1) = an(1,jy,1,lccna);
// WRF:13820
} else {
// WRF:13821
ccna(1) = cx(1,lc);
// WRF:13822
}
// WRF:13823
}
// WRF:13826
}
// WRF:13827
if((ipconc>=3)) {
// WRF:13828
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13829
cx(1,lr) = lake_max(an(1,jy,1,lnr),0.0f);
// WRF:13830
if((qx(1,lr)<=coeff_qxmin{}(lr))) {
// WRF:13832
} else if(((cx(1,lr)==0.0f)&&(qx(1,lr)<(3.0f*coeff_qxmin{}(lr))))) {
// WRF:13833
qx(1,lv) = (qx(1,lv)+qx(1,lr));
// WRF:13834
qx(1,lr) = 0.0f;
// WRF:13835
} else {
// WRF:13836
cx(1,lr) = lake_max(1.e-9f,cx(1,lr));
// WRF:13837
}
// WRF:13838
}
// WRF:13839
}
// WRF:13840
if((ipconc>=4)) {
// WRF:13841
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13842
cx(1,ls) = lake_max(an(1,jy,1,lns),0.0f);
// WRF:13843
if((qx(1,ls)<=coeff_qxmin{}(ls))) {
// WRF:13845
} else if(((cx(1,ls)==0.0f)&&(qx(1,ls)<(3.0f*coeff_qxmin{}(ls))))) {
// WRF:13846
qx(1,lv) = (qx(1,lv)+qx(1,ls));
// WRF:13847
qx(1,ls) = 0.0f;
// WRF:13848
} else {
// WRF:13849
cx(1,ls) = lake_max(1.e-9f,cx(1,ls));
// WRF:13851
if((ilimit>=coeff_ipc{}(ls))) {
// WRF:13852
tmp = lake_div(((coeff_xdn0{}(ls)*cx(1,ls))),((rho0(1)*qx(1,ls))));
// WRF:13853
tmp2 = lake_pow(((tmp*(3.14159f))),(lake_div(1.f,3.f)));
// WRF:13854
cnox = (cx(1,ls)*(tmp2));
// WRF:13855
if((cnox>(3.0f*coeff_cno{}(ls)))) {
// WRF:13856
cx(1,ls) = lake_div((3.0f*coeff_cno{}(ls)),tmp2);
// WRF:13857
}
// WRF:13858
}
// WRF:13859
}
// WRF:13860
}
// WRF:13861
}
// WRF:13862
if((ipconc>=5)) {
// WRF:13863
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13865
cx(1,lh) = lake_max(an(1,jy,1,lnh),0.0f);
// WRF:13866
if((qx(1,lh)<=coeff_qxmin{}(lh))) {
// WRF:13868
} else if(((cx(1,lh)==0.0f)&&(qx(1,lh)<(3.0f*coeff_qxmin{}(lh))))) {
// WRF:13869
qx(1,lv) = (qx(1,lv)+qx(1,lh));
// WRF:13870
qx(1,lh) = 0.0f;
// WRF:13871
} else {
// WRF:13872
cx(1,lh) = lake_max(1.e-9f,cx(1,lh));
// WRF:13873
if((ilimit>=coeff_ipc{}(lh))) {
// WRF:13874
tmp = lake_div(((coeff_xdn0{}(lh)*cx(1,lh))),((rho0(1)*qx(1,lh))));
// WRF:13875
tmp2 = lake_pow(((tmp*(3.14159f))),(lake_div(1.f,3.f)));
// WRF:13876
cnox = (cx(1,lh)*(tmp2));
// WRF:13877
if((cnox>(3.0f*coeff_cno{}(lh)))) {
// WRF:13878
cx(1,lh) = lake_div((3.0f*coeff_cno{}(lh)),tmp2);
// WRF:13879
}
// WRF:13880
}
// WRF:13881
}
// WRF:13884
}
// WRF:13887
}
// WRF:13889
if(((lhl>1)&&(ipconc>=5))) {
// WRF:13890
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13892
cx(1,lhl) = lake_max(an(1,jy,1,lnhl),0.0f);
// WRF:13893
if((qx(1,lhl)<=coeff_qxmin{}(lhl))) {
// WRF:13894
cx(1,lhl) = 0.0f;
// WRF:13895
} else if(((cx(1,lhl)==0.0f)&&(qx(1,lhl)<(3.0f*coeff_qxmin{}(lhl))))) {
// WRF:13896
qx(1,lv) = (qx(1,lv)+qx(1,lhl));
// WRF:13897
qx(1,lhl) = 0.0f;
// WRF:13898
} else {
// WRF:13899
cx(1,lhl) = lake_max(1.e-9f,cx(1,lhl));
// WRF:13900
if((ilimit>=coeff_ipc{}(lhl))) {
// WRF:13901
tmp = lake_div(((coeff_xdn0{}(lhl)*cx(1,lhl))),((rho0(1)*qx(1,lhl))));
// WRF:13902
tmp2 = lake_pow(((tmp*(3.14159f))),(lake_div(1.f,3.f)));
// WRF:13903
cnox = (cx(1,lhl)*(tmp2));
// WRF:13904
if((cnox>(3.0f*coeff_cno{}(lhl)))) {
// WRF:13905
cx(1,lhl) = lake_div((3.0f*coeff_cno{}(lhl)),tmp2);
// WRF:13906
}
// WRF:13907
}
// WRF:13908
}
// WRF:13911
}
// WRF:13912
}
// WRF:13917
if(ldovol) {
// WRF:13919
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=li;_s1<=lhab;++_s1) {vx(_s0,_s1) = 0.0f;}}
// WRF:13921
#pragma unroll
for(il=li;il<=lhab;il+=1) {
// WRF:13923
if((coeff_lvol{}(il)>=1)) {
// WRF:13925
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13926
vx(1,il) = lake_max(an(1,jy,1,coeff_lvol{}(il)),0.0f);
// WRF:13927
}
// WRF:13929
}
// WRF:13931
}
// WRF:13933
}
// WRF:13939
for(int _s0=1;_s0<=ngs;++_s0) {fhw(_s0) = 0.0f;}
// WRF:13940
for(int _s0=1;_s0<=ngs;++_s0) {fsw(_s0) = 0.0f;}
// WRF:13941
for(int _s0=1;_s0<=ngs;++_s0) {fhlw(_s0) = 0.0f;}
// WRF:13949
if((ipconc>=6)) {
// WRF:13950
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lr;_s1<=lhab;++_s1) {zx(_s0,_s1) = 0.0f;}}
// WRF:13951
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:13952
if((coeff_lz{}(il)>1)) {
// WRF:13953
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13954
zx(1,il) = lake_max(an(1,jy,1,coeff_lz{}(il)),0.0f);
// WRF:13955
}
// WRF:13956
}
// WRF:13957
}
// WRF:13959
}
// WRF:13961
if((ipconc>=6)) {
// WRF:13963
if((coeff_lz{}(lr)<1)) {
// WRF:13964
for(int _s0=1;_s0<=ngs;++_s0) {g1x(_s0,lr) = lake_div(((((6.0f+alphar))*((5.0f+alphar)))*((4.0f+alphar))),(((((3.0f+alphar))*((2.0f+alphar)))*((1.0f+alphar)))));}
// WRF:13968
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:13969
if(((cx(1,lr)>0.0f)&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:13971
vr = lake_div((rho0(1)*qx(1,lr)),((1000.f*cx(1,lr))));
// WRF:13972
if((lzr<1)) {
// WRF:13973
if((imurain==3)) {
// WRF:13974
zx(1,lr) = lake_div((((3.6476f*((rnu+2.0f)))*cx(1,lr))*lake_pow(vr,2)),((rnu+1.0f)));
// WRF:13975
} else {
// WRF:13976
zx(1,lr) = (((3.6476f*g1x(1,lr))*cx(1,lr))*lake_pow(vr,2));
// WRF:13977
}
// WRF:13978
}
// WRF:13980
}
// WRF:13981
}
// WRF:13982
}
// WRF:13984
}
// WRF:13987
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lc;_s1<=lhab;++_s1) {scx(_s0,_s1) = 0.0f;}}
// WRF:13991
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:13992
if((imurain==1)) {
// WRF:13993
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = alphar;}
// WRF:13994
} else if((imurain==3)) {
// WRF:13995
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lr) = coeff_xnu{}(lr);}
// WRF:13996
}
// WRF:13998
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,li) = coeff_xnu{}(li);}
// WRF:13999
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,lc) = coeff_xnu{}(lc);}
// WRF:14001
if((imusnow==1)) {
// WRF:14002
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,ls) = alphas;}
// WRF:14003
} else if((imusnow==3)) {
// WRF:14004
for(int _s0=1;_s0<=ngs;++_s0) {alpha(_s0,ls) = coeff_xnu{}(ls);}
// WRF:14005
}
// WRF:14007
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:14009
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:14010
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14011
if((il>=lg)) {alpha(1,il) = coeff_dnu{}(il);}
// WRF:14014
for(ic=lc;ic<=lhab;ic+=1) {
// WRF:14015
dab0lh(1,il,ic) = coeff_dab0{}(il,ic);
// WRF:14016
dab1lh(1,il,ic) = coeff_dab1{}(il,ic);
// WRF:14017
}
// WRF:14018
}
// WRF:14019
}
// WRF:14023
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:14024
for(int _s0=1;_s0<=ngs;++_s0) {da0lx(_s0,il) = coeff_da0{}(il);}
// WRF:14025
}
// WRF:14026
for(int _s0=1;_s0<=ngs;++_s0) {da0lh(_s0) = coeff_da0{}(lh);}
// WRF:14027
for(int _s0=1;_s0<=ngs;++_s0) {da0lr(_s0) = coeff_da0{}(lr);}
// WRF:14028
for(int _s0=1;_s0<=ngs;++_s0) {da1lr(_s0) = coeff_da1{}(lr);}
// WRF:14029
for(int _s0=1;_s0<=ngs;++_s0) {da0lc(_s0) = coeff_da0{}(lc);}
// WRF:14030
for(int _s0=1;_s0<=ngs;++_s0) {da1lc(_s0) = coeff_da1{}(lc);}
// WRF:14032
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:14034
if(((lzh<1)||(lzhl<1))) {
// WRF:14035
for(int _s0=1;_s0<=ngs;++_s0) {rzxhlh(_s0) = lake_div(rzhl,rz);}
// WRF:14036
} else if(((lzh>1)&&(lzhl>1))) {
// WRF:14037
for(int _s0=1;_s0<=ngs;++_s0) {rzxhlh(_s0) = 1.f;}
// WRF:14038
}
// WRF:14039
if((lzr>1)) {
// WRF:14040
for(int _s0=1;_s0<=ngs;++_s0) {rzxh(_s0) = 1.f;}
// WRF:14041
for(int _s0=1;_s0<=ngs;++_s0) {rzxhl(_s0) = 1.f;}
// WRF:14042
} else {
// WRF:14043
for(int _s0=1;_s0<=ngs;++_s0) {rzxh(_s0) = rz;}
// WRF:14044
for(int _s0=1;_s0<=ngs;++_s0) {rzxhl(_s0) = rzhl;}
// WRF:14045
}
// WRF:14047
if((((imurain==1)&&(imusnow==3))&&(lzr<1))) {
// WRF:14048
for(int _s0=1;_s0<=ngs;++_s0) {rzxs(_s0) = rzs;}
// WRF:14049
} else if(((imurain==imusnow)||(lzr>1))) {
// WRF:14050
for(int _s0=1;_s0<=ngs;++_s0) {rzxs(_s0) = 1.f;}
// WRF:14051
}
// WRF:14054
if((lhl>1)) {
// WRF:14055
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14056
da0lhl(1) = coeff_da0{}(lhl);
// WRF:14057
}
// WRF:14058
}
// WRF:14060
for(int _s0=1;_s0<=ngs;++_s0) {ventrx(_s0) = ventr;}
// WRF:14061
for(int _s0=1;_s0<=ngs;++_s0) {ventrxn(_s0) = ventrn;}
// WRF:14062
for(int _s0=1;_s0<=ngs;++_s0) {gf1palp(_s0) = gamma_sp((1.0f+alphar));}
// WRF:14067
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14069
ssi(1) = lake_div(qx(1,lv),qis(1));
// WRF:14070
ssw(1) = lake_div(qx(1,lv),qvs(1));
// WRF:14072
tsqr(1) = lake_pow(temg(1),2);
// WRF:14074
temgx(1) = lake_min(temg(1),313.15f);
// WRF:14075
temgx(1) = lake_max(temgx(1),233.15f);
// WRF:14076
felv(1) = (2500837.367f*lake_pow((lake_div(273.15f,temgx(1))),(((0.167f)+((3.67e-4f)*temgx(1))))));
// WRF:14078
temcgx(1) = lake_min(temg(1),273.15f);
// WRF:14079
temcgx(1) = lake_max(temcgx(1),223.15f);
// WRF:14080
temcgx(1) = (temcgx(1)-273.15f);
// WRF:14083
felf(1) = ((333690.6098f+((2030.61425f)*temcgx(1)))-((10.46708312f)*lake_pow(temcgx(1),2)));
// WRF:14085
fels(1) = (felv(1)+felf(1));
// WRF:14087
felvs(1) = (felv(1)*felv(1));
// WRF:14088
felss(1) = (fels(1)*fels(1));
// WRF:14090
if((eqtset<=1)) {
// WRF:14091
felvcp(1) = (felv(1)*cpi);
// WRF:14092
felscp(1) = (fels(1)*cpi);
// WRF:14093
felfcp(1) = (felf(1)*cpi);
// WRF:14094
} else {
// WRF:14099
tmp = ((qx(1,li)+qx(1,ls))+qx(1,lh));
// WRF:14100
if((lhl>1)) {tmp = (tmp+qx(1,lhl));}
// WRF:14101
if((lf>1)) {tmp = (tmp+qx(1,lf));}
// WRF:14102
cvm = (((cv+(cvv*qx(1,lv)))+(cpl*((qx(1,lc)+qx(1,lr)))))+(cpigb*(tmp)));
// WRF:14105
if((eqtset==2)) {
// WRF:14106
felvcp(1) = lake_div(((felv(1)-(rw*temg(1)))),cvm);
// WRF:14107
felscp(1) = lake_div(((fels(1)-(rw*temg(1)))),cvm);
// WRF:14108
felfcp(1) = lake_div(felf(1),cvm);
// WRF:14110
} else {
// WRF:14113
cpm = (((cp+(cpv*qx(1,lv)))+(cpl*((qx(1,lc)+qx(1,lr)))))+(cpigb*(tmp)));
// WRF:14115
rmm = (rd+(rw*qx(1,lv)));
// WRF:14117
felvcp(1) = lake_div(((lake_div((felv(1)*cv),(cp))-((rw*temg(1))*((1.0f-lake_div((rovcp*cpm),rmm)))))),cvm);
// WRF:14118
felscp(1) = lake_div(((lake_div((fels(1)*cv),(cp))-((rw*temg(1))*((1.0f-lake_div((rovcp*cpm),rmm)))))),cvm);
// WRF:14119
felfcp(1) = lake_div((felf(1)*cv),((cp*cvm)));
// WRF:14121
felvpi(1) = lake_div(((pi0(1)*rovcp)*((lake_div(felv(1),(temg(1)))-lake_div((rw*cpm),rmm)))),cvm);
// WRF:14122
felspi(1) = lake_div(((pi0(1)*rovcp)*((lake_div(fels(1),(temg(1)))-lake_div((rw*cpm),rmm)))),cvm);
// WRF:14123
felfpi(1) = ((pi0(1)*rovcp)*(lake_div(felf(1),((cvm*temg(1))))));
// WRF:14125
}
// WRF:14127
}
// WRF:14129
fgamw(1) = lake_div(felvcp(1),pi0(1));
// WRF:14130
fgams(1) = lake_div(felscp(1),pi0(1));
// WRF:14132
fcqv1(1) = ((4098.0258f*pi0(1))*fgamw(1));
// WRF:14133
fcqv2(1) = ((5807.6953f*pi0(1))*fgams(1));
// WRF:14134
fcc3(1) = lake_div(felfcp(1),pi0(1));
// WRF:14137
fwvdf(1) = (((2.11e-05f)*(lake_pow((lake_div(temg(1),tfr)),1.94f)))*(lake_div(101325.0f,(pres(1)))));
// WRF:14141
fadvisc(1) = ((advisc0*(lake_div(416.16f,((temg(1)+120.0f)))))*lake_pow((lake_div(temg(1),296.0f)),(1.5f)));
// WRF:14143
fakvisc(1) = (fadvisc(1)*rhoinv(1));
// WRF:14145
temcgx(1) = lake_min(temg(1),273.15f);
// WRF:14146
temcgx(1) = lake_max(temcgx(1),233.15f);
// WRF:14147
temcgx(1) = (temcgx(1)-273.15f);
// WRF:14148
fci(1) = (((2.118636f+(0.007371f*(temcgx(1)))))*(1.0e+03f));
// WRF:14150
if((temg(1)<273.15f)) {
// WRF:14151
temcgx(1) = lake_min(temg(1),273.15f);
// WRF:14152
temcgx(1) = lake_max(temcgx(1),233.15f);
// WRF:14153
temcgx(1) = (temcgx(1)-273.15f);
// WRF:14154
fcw(1) = ((4203.1548f+((1.30572e-2f)*(lake_pow(((temcgx(1)-35.f)),2))))+((1.60056e-5f)*(lake_pow(((temcgx(1)-35.f)),4))));
// WRF:14156
}
// WRF:14157
if((temg(1)>=273.15f)) {
// WRF:14158
temcgx(1) = lake_min(temg(1),308.15f);
// WRF:14159
temcgx(1) = lake_max(temcgx(1),273.15f);
// WRF:14160
temcgx(1) = (temcgx(1)-273.15f);
// WRF:14161
fcw(1) = (4243.1688f+((3.47104e-1f)*(lake_pow(temcgx(1),2))));
// WRF:14162
}
// WRF:14164
ftka(1) = lake_div((tka0*fadvisc(1)),advisc1);
// WRF:14165
fthdf(1) = ((ftka(1)*cpi)*rhoinv(1));
// WRF:14167
fschm(1) = (lake_div(fakvisc(1),fwvdf(1)));
// WRF:14168
fpndl(1) = (lake_div(fakvisc(1),fthdf(1)));
// WRF:14170
fai(1) = lake_div((lake_pow(fels(1),2)),(((ftka(1)*rw)*lake_pow(temg(1),2))));
// WRF:14171
fbi(1) = (lake_div(1.0f,(((rho0(1)*fwvdf(1))*qis(1)))));
// WRF:14172
fav(1) = lake_div((lake_pow(felv(1),2)),(((ftka(1)*rw)*lake_pow(temg(1),2))));
// WRF:14173
fbv(1) = (lake_div(1.0f,(((rho0(1)*fwvdf(1))*qvs(1)))));
// WRF:14175
kp1 = lake_min(nz,(1+1));
// WRF:14176
wvel(1) = .5f*(above_w+w(1,jgs,1));
// WRF:14180
}
// WRF:14189
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:14192
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14193
xdn(1,li) = coeff_xdn0{}(li);
// WRF:14194
xdn(1,lc) = coeff_xdn0{}(lc);
// WRF:14195
xdn(1,lr) = coeff_xdn0{}(lr);
// WRF:14196
xdn(1,ls) = coeff_xdn0{}(ls);
// WRF:14197
xdn(1,lh) = coeff_xdn0{}(lh);
// WRF:14198
if((coeff_lvol{}(ls)>1)) {
// WRF:14199
if(((vx(1,ls)>0.0f)&&(qx(1,ls)>coeff_qxmin{}(ls)))) {
// WRF:14200
xdn(1,ls) = lake_min(coeff_xdnmx{}(ls),lake_max(coeff_xdnmn{}(ls),lake_div((rho0(1)*qx(1,ls)),vx(1,ls))));
// WRF:14201
}
// WRF:14202
}
// WRF:14204
if((coeff_lvol{}(lh)>1)) {
// WRF:14205
if(((vx(1,lh)>0.0f)&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:14206
if(mixedphase) {
// WRF:14207
} else {
// WRF:14208
dnmx = coeff_xdnmx{}(lh);
// WRF:14209
}
// WRF:14210
xdn(1,lh) = lake_min(dnmx,lake_max(coeff_xdnmn{}(lh),lake_div((rho0(1)*qx(1,lh)),vx(1,lh))));
// WRF:14211
vx(1,lh) = lake_div((rho0(1)*qx(1,lh)),xdn(1,lh));
// WRF:14213
} else if(((vx(1,lh)==0.0f)&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:14215
vx(1,lh) = lake_div((rho0(1)*qx(1,lh)),xdn(1,lh));
// WRF:14217
}
// WRF:14218
}
// WRF:14221
if((lhl>1)) {
// WRF:14223
xdn(1,lhl) = coeff_xdn0{}(lhl);
// WRF:14224
xdntmp(1,lhl) = coeff_xdn0{}(lhl);
// WRF:14226
if((coeff_lvol{}(lhl)>1)) {
// WRF:14227
if(((vx(1,lhl)>0.0f)&&(qx(1,lhl)>coeff_qxmin{}(lhl)))) {
// WRF:14229
if((mixedphase&&(lhlw>1))) {
// WRF:14230
} else {
// WRF:14231
dnmx = coeff_xdnmx{}(lhl);
// WRF:14232
}
// WRF:14234
xdn(1,lhl) = lake_min(dnmx,lake_max(coeff_xdnmn{}(lhl),lake_div((rho0(1)*qx(1,lhl)),vx(1,lhl))));
// WRF:14235
vx(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),xdn(1,lhl));
// WRF:14236
xdntmp(1,lhl) = xdn(1,lhl);
// WRF:14238
} else if(((vx(1,lhl)==0.0f)&&(qx(1,lhl)>coeff_qxmin{}(lhl)))) {
// WRF:14240
vx(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),xdn(1,lhl));
// WRF:14242
}
// WRF:14243
}
// WRF:14245
}
// WRF:14248
}
// WRF:14250
if(((ipconc==5)&&(imydiagalpha==2))) {
// WRF:14252
cwchtmp = lake_pow((((((3.f+coeff_dnu{}(lh)))*((2.f+coeff_dnu{}(lh))))*((1.0f+coeff_dnu{}(lh))))),(lake_div((-1.f),3.f)));
// WRF:14254
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14256
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(cx(1,lr)>cxmin))) {
// WRF:14257
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*cx(1,lr))));
// WRF:14258
xdia(1,lr,3) = lake_pow((((xv(1,lr)*6.0f)*cwc1)),(lake_div(1.f,3.f)));
// WRF:14263
tmp = (4.f+alphar);
// WRF:14264
i = ((int)((dgami*(tmp))));
// WRF:14265
del = (tmp-(dgam*i));
// WRF:14266
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14268
tmp = (1.f+alphar);
// WRF:14269
i = ((int)((dgami*(tmp))));
// WRF:14270
del = (tmp-(dgam*i));
// WRF:14271
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14273
tmp = ((lake_pow((lake_div(x,y)),(lake_div(1.f,3.f)))*xdia(1,lr,3))*cwchtmp);
// WRF:14275
alpha(1,lr) = lake_min(15.f,((11.8f*lake_pow((((1000.f*tmp)-0.7f)),2))+2.f));
// WRF:14276
}
// WRF:14277
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(cx(1,lh)>cxmin))) {
// WRF:14279
xv(1,lh) = lake_div((rho0(1)*qx(1,lh)),((xdn(1,lh)*cx(1,lh))));
// WRF:14280
xdia(1,lh,3) = lake_pow((((xv(1,lh)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:14284
tmp = (4.f+coeff_dnu{}(lh));
// WRF:14285
i = ((int)((dgami*(tmp))));
// WRF:14286
del = (tmp-(dgam*i));
// WRF:14287
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14289
tmp = (1.f+coeff_dnu{}(lh));
// WRF:14290
i = ((int)((dgami*(tmp))));
// WRF:14291
del = (tmp-(dgam*i));
// WRF:14292
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14294
tmp = ((lake_pow((lake_div(x,y)),(lake_div(1.f,3.f)))*xdia(1,lh,3))*cwchtmp);
// WRF:14296
alpha(1,lh) = lake_min(15.f,((11.8f*lake_pow((((1000.f*tmp)-0.7f)),2))+2.f));
// WRF:14300
il = lh;
// WRF:14301
for(ic=lc;ic<=(lh-1);ic+=1) {
// WRF:14302
i = ((int)round((alpha(1,il)*dqiacralphainv)));
// WRF:14303
if(((((ic==lc)||(ic==li))||(ic==ls))||(((ic==lr)&&(imurain==3))))) {
// WRF:14304
alp = (((3.f*alpha(1,ic))+2.f));
// WRF:14305
j = ((int)round(((((3.f*alpha(1,ic))+2.f))*dqiacralphainv)));
// WRF:14306
} else {
// WRF:14307
alp = alpha(1,ic);
// WRF:14308
j = ((int)round((alpha(1,ic)*dqiacralphainv)));
// WRF:14309
}
// WRF:14311
dab0lh(1,ic,il) = dab0lu(j,i,ic,il);
// WRF:14312
dab1lh(1,ic,il) = dab1lu(j,i,ic,il);
// WRF:14313
dab0lh(1,il,ic) = dab0lu(i,j,il,ic);
// WRF:14314
dab1lh(1,il,ic) = dab1lu(i,j,il,ic);
// WRF:14315
}
// WRF:14316
}
// WRF:14319
if((lhl>0)) {
// WRF:14320
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(cx(1,lhl)>cxmin))) {
// WRF:14321
xv(1,lhl) = lake_div((rho0(1)*qx(1,lhl)),((xdn(1,lhl)*cx(1,lhl))));
// WRF:14322
xdia(1,lhl,3) = lake_pow((((xv(1,lhl)*6.f)*piinv)),(lake_div(1.f,3.f)));
// WRF:14323
if((xdia(1,lhl,3)<0.008f)) {
// WRF:14324
alpha(1,lhl) = lake_min(alphamax,((c1hl*tanhf((c2hl*(((xdia(1,lhl,3)*1000.f)-c3hl)))))+c4hl));
// WRF:14325
} else {
// WRF:14326
alpha(1,lhl) = lake_min(alphamax,(((c5hl*xdia(1,lhl,3))*1000.f)+c6hl));
// WRF:14327
}
// WRF:14329
il = lhl;
// WRF:14330
for(ic=lc;ic<=(lh-1);ic+=1) {
// WRF:14331
i = ((int)round((alpha(1,il)*dqiacralphainv)));
// WRF:14332
if(((((ic==lc)||(ic==li))||(ic==ls))||(((ic==lr)&&(imurain==3))))) {
// WRF:14333
alp = (((3.f*alpha(1,ic))+2.f));
// WRF:14334
j = ((int)round(((((3.f*alpha(1,ic))+2.f))*dqiacralphainv)));
// WRF:14335
} else {
// WRF:14336
alp = alpha(1,ic);
// WRF:14337
j = ((int)round((alpha(1,ic)*dqiacralphainv)));
// WRF:14338
}
// WRF:14340
dab0lh(1,ic,il) = dab0lu(j,i,ic,il);
// WRF:14341
dab1lh(1,ic,il) = dab1lu(j,i,ic,il);
// WRF:14342
dab0lh(1,il,ic) = dab0lu(i,j,il,ic);
// WRF:14343
dab1lh(1,il,ic) = dab1lu(i,j,il,ic);
// WRF:14344
}
// WRF:14346
}
// WRF:14347
}
// WRF:14351
}
// WRF:14352
}
// WRF:14355
if((imurain==3)) {
// WRF:14356
if((lzr>1)) {
// WRF:14357
alphashr = 0.0f;
// WRF:14358
alphamlr = lake_div((-2.0f),3.0f);
// WRF:14359
alphasmlr = lake_div((-2.0f),3.0f);
// WRF:14360
} else {
// WRF:14361
alphashr = coeff_xnu{}(lr);
// WRF:14362
alphamlr = coeff_xnu{}(lr);
// WRF:14363
alphasmlr = coeff_xnu{}(lr);
// WRF:14364
}
// WRF:14367
massfacshr = (lake_div(lake_pow(((2.f+(3.f*((1.f+alphashr))))),3),((3.f*((1.f+alphashr))))));
// WRF:14368
massfacmlr = (lake_div(lake_pow(((2.f+(3.f*((1.f+alphamlr))))),3),((3.f*((1.f+alphamlr))))));
// WRF:14369
} else if((imurain==1)) {
// WRF:14370
if((lzr>1)) {
// WRF:14371
alphashr = 4.0f;
// WRF:14372
alphamlr = 4.0f;
// WRF:14373
alphasmlr = alphasmlr0;
// WRF:14374
} else {
// WRF:14375
alphashr = alphar;
// WRF:14376
alphamlr = alphar;
// WRF:14377
alphasmlr = alphar;
// WRF:14378
}
// WRF:14381
massfacshr = lake_div(lake_pow(((3.0f+alphashr)),3),(((((3.f+alphashr))*((2.f+alphashr)))*((1.f+alphashr)))));
// WRF:14382
massfacmlr = lake_div(lake_pow(((3.0f+alphamlr)),3),(((((3.f+alphamlr))*((2.f+alphamlr)))*((1.f+alphamlr)))));
// WRF:14383
}
// WRF:14387
g1shr = 1.0f;
// WRF:14388
g1mlr = 1.0f;
// WRF:14389
g1smlr = 1.0f;
// WRF:14393
if((ipconc>=6)) {
// WRF:14396
if(((ipconc>=6)&&(imurain==3))) {
// WRF:14397
il = lr;
// WRF:14398
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14400
g1x(1,il) = lake_div(((alpha(1,lr)+2.0f)),(((alpha(1,lr)+1.0f))));
// WRF:14401
}
// WRF:14402
}
// WRF:14404
if((lzr>1)) {
// WRF:14405
if((imurain==3)) {
// WRF:14406
g1shr = lake_div(((alphashr+2.0f)),(((alphashr+1.0f))));
// WRF:14407
g1mlr = lake_div(((alphamlr+2.0f)),(((alphamlr+1.0f))));
// WRF:14408
g1smlr = lake_div(((alphasmlr+2.0f)),(((alphasmlr+1.0f))));
// WRF:14409
} else if((imurain==1)) {
// WRF:14412
g1shr = lake_div(((((6.0f+alphashr))*((5.0f+alphashr)))*((4.0f+alphashr))),(((((3.0f+alphashr))*((2.0f+alphashr)))*((1.0f+alphashr)))));
// WRF:14416
g1mlr = lake_div(((((6.0f+alphamlr))*((5.0f+alphamlr)))*((4.0f+alphamlr))),(((((3.0f+alphamlr))*((2.0f+alphamlr)))*((1.0f+alphamlr)))));
// WRF:14418
g1smlr = lake_div(((((6.0f+alphasmlr))*((5.0f+alphasmlr)))*((4.0f+alphasmlr))),(((((3.0f+alphasmlr))*((2.0f+alphasmlr)))*((1.0f+alphasmlr)))));
// WRF:14420
}
// WRF:14421
}
// WRF:14423
if(((lzr>1)&&(imurain==3))) {
// WRF:14427
il = lr;
// WRF:14428
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14431
if((((iresetmoments==1)||(iresetmoments==il))||(iresetmoments==(-1)))) {
// WRF:14432
if((zx(1,il)<=zxmin)) {
// WRF:14434
qx(1,il) = 0.0f;
// WRF:14435
cx(1,il) = 0.0f;
// WRF:14436
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14437
an(1,jgs,1,il) = qx(1,il);
// WRF:14438
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14439
} else if(((iresetmoments==(-1))&&(qx(1,il)<coeff_qxmin{}(il)))) {
// WRF:14440
zx(1,il) = 0.0f;
// WRF:14441
cx(1,il) = 0.0f;
// WRF:14442
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14444
qx(1,il) = 0.0f;
// WRF:14445
an(1,jgs,1,il) = qx(1,il);
// WRF:14446
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14447
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14449
} else if(((cx(1,il)<=cxmin)&&(iresetmoments!=(-1)))) {
// WRF:14451
qx(1,lv) = (qx(1,lv)+qx(1,il));
// WRF:14452
zx(1,lr) = 0.0f;
// WRF:14453
qx(1,lr) = 0.0f;
// WRF:14454
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,lr));
// WRF:14455
an(1,jgs,1,lr) = qx(1,lr);
// WRF:14456
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:14457
}
// WRF:14458
}
// WRF:14460
if(((false&&(zx(1,il)<=zxmin))&&(cx(1,il)<=cxmin))) {
// WRF:14461
zx(1,il) = 0.0f;
// WRF:14462
cx(1,il) = 0.0f;
// WRF:14463
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14465
qx(1,il) = 0.0f;
// WRF:14466
an(1,jgs,1,il) = qx(1,il);
// WRF:14467
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14468
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14469
}
// WRF:14471
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:14473
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*lake_max(1.0e-11f,cx(1,lr)))));
// WRF:14474
if((xv(1,lr)>coeff_xvmx{}(lr))) {
// WRF:14477
} else if((xv(1,lr)<coeff_xvmn{}(lr))) {
// WRF:14478
xv(1,lr) = coeff_xvmn{}(lr);
// WRF:14479
cx(1,lr) = lake_div((rho0(1)*qx(1,lr)),((coeff_xvmn{}(lr)*xdn(1,lr))));
// WRF:14480
}
// WRF:14482
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:14484
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14485
z = zx(1,il);
// WRF:14486
qr = qx(1,il);
// WRF:14487
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),((z*lake_pow(xdn(1,lr),2))));
// WRF:14489
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>0.0f))) {
// WRF:14491
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14492
chw = cx(1,il);
// WRF:14493
qr = qx(1,il);
// WRF:14494
zx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),((lake_pow(xdn(1,lr),2)*chw)));
// WRF:14495
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:14497
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:14501
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:14502
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14504
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14505
z = zx(1,il);
// WRF:14506
qr = qx(1,il);
// WRF:14507
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:14508
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14509
}
// WRF:14511
if((zx(1,lr)>0.0f)) {
// WRF:14512
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((1000.f*cx(1,lr))));
// WRF:14513
vr = xv(1,lr);
// WRF:14514
qr = qx(1,lr);
// WRF:14515
nrx = cx(1,lr);
// WRF:14516
z = zx(1,lr);
// WRF:14522
if((z>0.0f)) {
// WRF:14524
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:14525
for(i=1;i<=20;i+=1) {
// WRF:14526
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:14527
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:14528
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:14529
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:14530
}
// WRF:14533
if((((xv(1,il)>coeff_xvmx{}(il))||(((ioldlimiter>=2)&&(xv(1,il)>lake_div(coeff_xvmx{}(il),8.f))))))) {
// WRF:14534
tmp = cx(1,il);
// WRF:14535
if((ioldlimiter>=2)) {
// WRF:14536
x = lake_pow((lake_div(((6.f*rho0(1))*qx(1,il)),(((pi*xdn(1,il))*cx(1,il))))),(lake_div(1.f,3.f)));
// WRF:14537
x1 = lake_max(0.0e-3f,(x-3.0e-3f));
// WRF:14538
x2 = lake_max(0.5f,lake_div(x,6.0e-3f));
// WRF:14539
x3 = lake_pow(x2,3);
// WRF:14540
cx(1,il) = (cx(1,il)*lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:14541
xv(1,il) = lake_div(xv(1,il),lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:14542
} else {
// WRF:14543
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:14544
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:14545
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:14546
}
// WRF:14550
if((tmp<cx(1,il))) {
// WRF:14552
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14553
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:14554
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14556
vr = xv(1,lr);
// WRF:14557
qr = qx(1,lr);
// WRF:14558
nrx = cx(1,lr);
// WRF:14559
z = zx(1,lr);
// WRF:14563
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:14564
for(i=1;i<=20;i+=1) {
// WRF:14565
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:14566
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:14567
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:14568
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:14569
}
// WRF:14572
}
// WRF:14573
}
// WRF:14579
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14580
if((true&&((((alpha(1,il)<=rnumin)||(alp==rnumin))||(alp==rnumax))))) {
// WRF:14582
if((rescale_high_alpha&&(alp>=(rnumax-0.01f)))) {
// WRF:14583
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(1.f,(xdn(1,il)))),2));
// WRF:14584
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14586
} else if((rescale_low_alphar&&(alp<=rnumin))) {
// WRF:14587
z = lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:14588
zx(1,il) = z;
// WRF:14589
an(1,jy,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14590
}
// WRF:14591
}
// WRF:14597
if((alp>=(rnumax-0.01f))) {
// WRF:14600
g1x(1,il) = lake_div(((lake_pow(((pi*xdn(1,il))),2)*zx(1,il))*cx(1,il)),(lake_pow((((6.f*rho0(1))*qx(1,il))),2)));
// WRF:14601
} else {
// WRF:14602
g1x(1,il) = g1;
// WRF:14603
}
// WRF:14605
tmp = (alpha(1,lr)+lake_div(4.f,3.f));
// WRF:14606
i = ((int)((dgami*(tmp))));
// WRF:14607
del = (tmp-(dgam*i));
// WRF:14608
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14610
tmp = (alpha(1,lr)+1.f);
// WRF:14611
i = ((int)((dgami*(tmp))));
// WRF:14612
del = (tmp-(dgam*i));
// WRF:14613
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14615
gf1palp(1) = y;
// WRF:14618
ventrx(1) = lake_div(x,((y*lake_pow(((alpha(1,lr)+1.f)),(lake_div(1.f,3.f))))));
// WRF:14620
if(((imurain==3)&&(izwisventr==2))) {
// WRF:14622
tmp = ((alpha(1,lr)+1.5f)+lake_div(br,6.f));
// WRF:14623
i = ((int)((dgami*(tmp))));
// WRF:14624
del = (tmp-(dgam*i));
// WRF:14625
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14628
ventrxn(1) = lake_div(x,((y*lake_pow(((alpha(1,lr)+1.f)),((lake_div(((1.f+br)),6.f)+lake_div(1.f,3.f)))))));
// WRF:14642
}
// WRF:14644
}
// WRF:14645
}
// WRF:14647
}
// WRF:14649
}
// WRF:14651
}
// WRF:14653
}
// WRF:14656
if((ipconc>=6)) {
// WRF:14658
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:14661
if((((!(((il==lr)&&(imurain==3)))))&&(((((il==lr)||(il==lh))||(il==lhl))||(il==lf))))) {
// WRF:14662
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14663
g1x(1,il) = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:14665
}
// WRF:14666
}
// WRF:14668
if(((coeff_lz{}(il)>1)&&((!(((il==lr)&&(imurain==3))))))) {
// WRF:14670
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14673
if((((iresetmoments==1)||(iresetmoments==il))||(iresetmoments==(-1)))) {
// WRF:14674
if((zx(1,il)<=zxmin)) {
// WRF:14676
qx(1,il) = 0.0f;
// WRF:14677
cx(1,il) = 0.0f;
// WRF:14678
zx(1,il) = 0.0f;
// WRF:14679
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14680
an(1,jgs,1,il) = qx(1,il);
// WRF:14681
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14682
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14683
} else if(((iresetmoments==(-1))&&(qx(1,il)<coeff_qxmin{}(il)))) {
// WRF:14684
zx(1,il) = 0.0f;
// WRF:14685
cx(1,il) = 0.0f;
// WRF:14686
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14688
qx(1,il) = 0.0f;
// WRF:14689
an(1,jgs,1,il) = qx(1,il);
// WRF:14690
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14691
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14693
} else if(((cx(1,il)<=cxmin)&&(iresetmoments!=(-1)))) {
// WRF:14694
qx(1,lv) = (qx(1,lv)+qx(1,il));
// WRF:14695
zx(1,il) = 0.0f;
// WRF:14696
cx(1,il) = 0.0f;
// WRF:14697
qx(1,il) = 0.0f;
// WRF:14698
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14699
an(1,jgs,1,il) = qx(1,il);
// WRF:14700
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14701
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14702
}
// WRF:14703
}
// WRF:14705
if(((zx(1,il)<=zxmin)&&(cx(1,il)<=cxmin))) {
// WRF:14706
zx(1,il) = 0.0f;
// WRF:14707
cx(1,il) = 0.0f;
// WRF:14708
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:14710
qx(1,il) = 0.0f;
// WRF:14711
an(1,jgs,1,il) = qx(1,il);
// WRF:14712
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14713
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14714
}
// WRF:14716
if((qx(1,il)>coeff_qxmin{}(il))) {
// WRF:14718
xv(1,il) = lake_div((rho0(1)*qx(1,il)),((xdn(1,il)*lake_max(1.0e-9f,cx(1,il)))));
// WRF:14719
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:14721
if((xv(1,il)<coeff_xvmn{}(il))) {
// WRF:14722
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:14723
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:14724
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:14725
}
// WRF:14727
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:14729
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:14731
z = zx(1,il);
// WRF:14732
qr = qx(1,il);
// WRF:14734
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6.f*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:14736
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>cxmin))) {
// WRF:14740
chw = cx(1,il);
// WRF:14741
qr = qx(1,il);
// WRF:14744
g1 = lake_div(((((6.0f+alphamax))*((5.0f+alphamax)))*((4.0f+alphamax))),(((((3.0f+alphamax))*((2.0f+alphamax)))*((1.0f+alphamax)))));
// WRF:14746
zx(1,il) = lake_max((zxmin*1.1f),lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6*qr)),2)),((chw*lake_pow(((pi*xdn(1,il))),2)))));
// WRF:14747
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14749
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:14753
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:14754
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14756
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:14758
z = zx(1,il);
// WRF:14759
qr = qx(1,il);
// WRF:14761
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6.f*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:14762
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14763
} else {
// WRF:14765
chw = cx(1,il);
// WRF:14766
qr = qx(1,il);
// WRF:14767
z = zx(1,il);
// WRF:14769
if((zx(1,il)>0.f)) {
// WRF:14772
rdi = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((rho0(1)*qr)),2)));
// WRF:14776
alp = (lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:14779
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14781
if(newton) {
// WRF:14782
for(i=1;i<=10;i+=1) {
// WRF:14783
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:14784
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14785
alp = (alp+lake_div(((galpha(alp)-rdi)),dgalpha(alp)));
// WRF:14786
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14787
}
// WRF:14789
} else {
// WRF:14790
for(i=1;i<=10;i+=1) {
// WRF:14792
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:14793
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14796
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:14799
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14800
}
// WRF:14801
}
// WRF:14805
if((imaxdiaopt==1)) {
// WRF:14806
xvbarmax = coeff_xvmx{}(il);
// WRF:14807
} else if((imaxdiaopt==2)) {
// WRF:14808
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((3.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:14809
} else if((imaxdiaopt==3)) {
// WRF:14810
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((4.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:14811
} else {
// WRF:14812
xvbarmax = coeff_xvmx{}(il);
// WRF:14813
}
// WRF:14815
if(((xv(1,il)>xvbarmax)||((((il==lr)&&(ioldlimiter>=2))&&(xv(1,il)>lake_div(coeff_xvmx{}(il),8.f)))))) {
// WRF:14816
tmp = cx(1,il);
// WRF:14817
if(((ioldlimiter>=2)&&(il==lr))) {
// WRF:14818
x = lake_pow((lake_div(((6.f*rho0(1))*qx(1,il)),(((pi*xdn(1,il))*cx(1,il))))),(lake_div(1.f,3.f)));
// WRF:14819
x1 = lake_max(0.0e-3f,(x-3.0e-3f));
// WRF:14820
x2 = lake_max(0.5f,lake_div(x,6.0e-3f));
// WRF:14821
x3 = lake_pow(x2,3);
// WRF:14822
cx(1,il) = (cx(1,il)*lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:14823
xv(1,il) = lake_div(xv(1,il),lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:14824
} else {
// WRF:14825
xv(1,il) = lake_min(xvbarmax,lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:14826
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:14827
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:14828
}
// WRF:14829
if((tmp<cx(1,il))) {
// WRF:14830
g1 = lake_div((((36.f*((6.0f+alpha(1,il))))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),((((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))*lake_pow(pi,2))));
// WRF:14832
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:14833
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:14835
chw = cx(1,il);
// WRF:14836
qr = qx(1,il);
// WRF:14837
z = zx(1,il);
// WRF:14839
rdi = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((rho0(1)*qr)),2)));
// WRF:14840
alp = (lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:14842
for(i=1;i<=10;i+=1) {
// WRF:14843
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:14844
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14845
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:14847
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:14848
}
// WRF:14851
}
// WRF:14852
}
// WRF:14858
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:14861
if((((lrescalelow(il)||rescale_high_alpha))&&((((alpha(1,il)<=alphamin)||(alp==alphamin))||(alp==alphamax))))) {
// WRF:14866
if((rescale_high_alpha&&(alp>=(alphamax-0.01f)))) {
// WRF:14867
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:14868
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:14870
} else if((((lrescalelow(il)&&(alp<=alphamin))&&(!(((il==lh)&&(icvhl2h>0)))))&&(!(((il==lr)&&(!rescale_low_alphar)))))) {
// WRF:14872
wtest = false;
// WRF:14873
if((irescalerainopt==0)) {
// WRF:14874
wtest = false;
// WRF:14875
} else if((irescalerainopt==1)) {
// WRF:14876
wtest = (qx(1,lc)>coeff_qxmin{}(lc));
// WRF:14877
} else if((irescalerainopt==2)) {
// WRF:14878
wtest = ((qx(1,lc)>coeff_qxmin{}(lc))&&(wvel(1)<rescale_wthresh));
// WRF:14879
} else if((irescalerainopt==3)) {
// WRF:14880
wtest = (((temcg(1)>rescale_tempthresh)&&(qx(1,lc)>coeff_qxmin{}(lc)))&&(wvel(1)<rescale_wthresh));
// WRF:14881
}
// WRF:14883
if(((il==lr)&&(wtest))) {
// WRF:14887
chw = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:14888
cx(1,il) = chw;
// WRF:14889
an(1,jy,1,coeff_ln{}(il)) = chw;
// WRF:14890
} else {
// WRF:14893
z1 = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),chw);
// WRF:14894
z = (z1*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:14895
zx(1,il) = z;
// WRF:14896
an(1,jy,1,coeff_lz{}(il)) = z;
// WRF:14897
}
// WRF:14898
}
// WRF:14899
}
// WRF:14908
if((alp>=(alphamax-0.5f))) {
// WRF:14911
g1x(1,il) = lake_div(((lake_pow(((pi*xdn(1,il))),2)*zx(1,il))*cx(1,il)),(lake_pow((((6.f*rho0(1))*qx(1,il))),2)));
// WRF:14912
} else {
// WRF:14913
g1x(1,il) = g1;
// WRF:14914
}
// WRF:14916
}
// WRF:14925
}
// WRF:14927
if((il==lr)) {
// WRF:14943
tmp = (alpha(1,lr)+1.f);
// WRF:14944
i = ((int)((dgami*(tmp))));
// WRF:14945
del = (tmp-(dgam*i));
// WRF:14946
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14948
gf1palp(1) = y;
// WRF:14950
if((iferwisventr==2)) {
// WRF:14951
tmp = ((alpha(1,lr)+2.5f)+lake_div(br,2.f));
// WRF:14952
i = ((int)((dgami*(tmp))));
// WRF:14953
del = (tmp-(dgam*i));
// WRF:14954
x = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:14958
ventrxn(1) = lake_div(x,y);
// WRF:14960
}
// WRF:14962
}
// WRF:14965
} else {
// WRF:14973
}
// WRF:14978
}
// WRF:14987
if((!(((il==lr)&&(imurain==3))))) {
// WRF:14989
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:14990
if((qx(1,il)>coeff_qxmin{}(il))) {
// WRF:14991
xnutmp = lake_div(((alpha(1,il)-2.f)),3.f);
// WRF:14994
for(ic=lc;ic<=(lh-1);ic+=1) {
// WRF:14995
if(((il!=ic)&&(qx(1,ic)>coeff_qxmin{}(ic)))) {
// WRF:14996
xnuc = coeff_xnu{}(ic);
// WRF:14997
if(((ic==lc)&&(idiagnosecnu>0))) {xnuc = alpha(1,lc);}
// WRF:14998
if((((il!=lr)&&(ic==lr))&&(lzr>1))) {
// WRF:14999
if((imurain==3)) {
// WRF:15000
xnuc = alpha(1,lr);
// WRF:15001
} else {
// WRF:15002
xnuc = lake_div(((alpha(1,lr)-2.f)),3.f);
// WRF:15003
}
// WRF:15004
}
// WRF:15006
if(false) {
// WRF:15007
dab0lh(1,ic,il) = delabk(coeff_bb{}(ic),coeff_bb{}(il),xnuc,xnutmp,coeff_xmu{}(ic),coeff_xmu{}(il),0);
// WRF:15008
dab1lh(1,ic,il) = delabk(coeff_bb{}(ic),coeff_bb{}(il),xnuc,xnutmp,coeff_xmu{}(ic),coeff_xmu{}(il),1);
// WRF:15009
dab0lh(1,il,ic) = delabk(coeff_bb{}(il),coeff_bb{}(ic),xnutmp,xnuc,coeff_xmu{}(il),coeff_xmu{}(ic),0);
// WRF:15010
dab1lh(1,il,ic) = delabk(coeff_bb{}(il),coeff_bb{}(ic),xnutmp,xnuc,coeff_xmu{}(il),coeff_xmu{}(ic),1);
// WRF:15011
} else {
// WRF:15012
i = ((int)round((alpha(1,il)*dqiacralphainv)));
// WRF:15013
if(((((ic==lc)||(ic==li))||(ic==ls))||(((ic==lr)&&(imurain==3))))) {
// WRF:15014
alp = (((3.f*alpha(1,ic))+2.f));
// WRF:15015
j = ((int)round(((((3.f*alpha(1,ic))+2.f))*dqiacralphainv)));
// WRF:15016
} else {
// WRF:15017
alp = alpha(1,ic);
// WRF:15018
j = ((int)round((alpha(1,ic)*dqiacralphainv)));
// WRF:15019
}
// WRF:15021
dab0lh(1,ic,il) = dab0lu(j,i,ic,il);
// WRF:15022
dab1lh(1,ic,il) = dab1lu(j,i,ic,il);
// WRF:15023
dab0lh(1,il,ic) = dab0lu(i,j,il,ic);
// WRF:15024
dab1lh(1,il,ic) = dab1lu(i,j,il,ic);
// WRF:15035
if((false&&(ny<=2))) {
// WRF:15036
; /* native diagnostic only */
// WRF:15037
; /* native diagnostic only */
// WRF:15038
; /* native diagnostic only */
// WRF:15039
; /* native diagnostic only */
// WRF:15040
; /* native diagnostic only */
// WRF:15041
; /* native diagnostic only */
// WRF:15042
; /* native diagnostic only */
// WRF:15044
}
// WRF:15046
}
// WRF:15048
}
// WRF:15049
}
// WRF:15053
da0lx(1,il) = delbk(coeff_bb{}(il),xnutmp,coeff_xmu{}(il),0);
// WRF:15054
if((il==lh)) {
// WRF:15055
da0lh(1) = delbk(coeff_bb{}(il),xnutmp,coeff_xmu{}(il),0);
// WRF:15056
if((lzr>1)) {
// WRF:15057
rzxh(1) = 1.f;
// WRF:15058
} else {
// WRF:15059
rzxh(1) = lake_div(((((((4.f+alpha(1,il)))*((5.f+alpha(1,il))))*((6.f+alpha(1,il))))*((1.f+coeff_xnu{}(lr))))),((((((1.f+alpha(1,il)))*((2.f+alpha(1,il))))*((3.f+alpha(1,il))))*((2.f+coeff_xnu{}(lr))))));
// WRF:15061
}
// WRF:15063
if((lzhl<1)) {
// WRF:15064
rzxhlh(1) = lake_div(rzxhl(1),(lake_div(((((((4.f+alpha(1,il)))*((5.f+alpha(1,il))))*((6.f+alpha(1,il))))*((1.f+coeff_xnu{}(lr))))),((((((1.f+alpha(1,il)))*((2.f+alpha(1,il))))*((3.f+alpha(1,il))))*((2.f+coeff_xnu{}(lr))))))));
// WRF:15066
}
// WRF:15067
} else if((il==lhl)) {
// WRF:15068
da0lhl(1) = delbk(coeff_bb{}(il),xnutmp,coeff_xmu{}(il),0);
// WRF:15069
if((lzr>1)) {
// WRF:15070
rzxhl(1) = 1.f;
// WRF:15071
} else {
// WRF:15072
rzxhl(1) = lake_div(((((((4.0f+alpha(1,il)))*((5.f+alpha(1,il))))*((6.f+alpha(1,il))))*((1.f+coeff_xnu{}(lr))))),((((((1.f+alpha(1,il)))*((2.f+alpha(1,il))))*((3.f+alpha(1,il))))*((2.f+coeff_xnu{}(lr))))));
// WRF:15074
}
// WRF:15075
} else if((il==lr)) {
// WRF:15076
xnutmp = lake_div(((alpha(1,il)-2.f)),3.f);
// WRF:15077
da0lr(1) = delbk(coeff_bb{}(il),xnutmp,coeff_xmu{}(il),0);
// WRF:15078
da1lr(1) = delbk(coeff_bb{}(il),xnutmp,coeff_xmu{}(il),1);
// WRF:15079
}
// WRF:15081
}
// WRF:15082
}
// WRF:15084
}
// WRF:15088
}
// WRF:15090
}
// WRF:15092
}
// WRF:15099
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15100
kp1 = lake_min(nz,(1+1));
// WRF:15105
wvelkm1(1) = .5f*(w(1,jgs,1)+below_w);
// WRF:15107
cninm(1) = below_in;
// WRF:15108
cnina(1) = t7(1,jgs,1);
// WRF:15109
cninp(1) = above_in;
// WRF:15110
}
// WRF:15126
infdo = 1;
// WRF:15127
if((rimdenvwgt>0)) {infdo = 1;}
// WRF:15129
setvtz_point(ngscnt,qx,qxmin,qxw,cx,rho0,rhovt,xdia,cno,cnostmp,xmas,vtxbar,xdn,xvmn,xvmx,xv,cdx,cdxgs,ipconc,ndebug,ngs,nz,kgs,fadvisc,cwmasn,cwmasx,cwradn,cnina,cimn,cimx,itype1,itype2,temcg,infdo,alpha,0,axx,bxx);
// WRF:15137
if((lwsm6&&(ipconc==0))) {
// WRF:15138
tmp = lake_max(coeff_qxmin{}(lh),coeff_qxmin{}(ls));
// WRF:15139
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15140
total = (qx(1,lh)+qx(1,ls));
// WRF:15141
if((total>tmp)) {
// WRF:15142
vt2ave(1) = lake_div((((qx(1,lh)*vtxbar(1,lh,1))+(qx(1,ls)*vtxbar(1,ls,1)))),total);
// WRF:15143
} else {
// WRF:15144
vt2ave(1) = 0.0f;
// WRF:15145
}
// WRF:15146
}
// WRF:15147
}
// WRF:15153
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:15154
if((ipconc<1)) {
// WRF:15155
for(int _s0=1;_s0<=ngscnt;++_s0) {cina(_s0) = cx((1+(_s0-1)),li);}
// WRF:15156
}
// WRF:15157
if((ipconc<5)) {
// WRF:15158
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15161
if((ipconc<3)) {
// WRF:15163
if((qx(1,lr)>coeff_qxmin{}(lh))) {
// WRF:15166
}
// WRF:15167
}
// WRF:15169
if((ipconc<4)) {
// WRF:15172
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:15175
}
// WRF:15176
}
// WRF:15178
if((ipconc<5)) {
// WRF:15182
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:15186
}
// WRF:15188
}
// WRF:15190
}
// WRF:15191
}
// WRF:15193
if((ipconc>=2)) {
// WRF:15194
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15196
rb(1) = ((0.5f*xdia(1,lc,1))*lake_pow((lake_div(1.f,((1.f+alpha(1,lc))))),(lake_div(1.f,6.f))));
// WRF:15197
xl2p(1) = lake_max(0.0e0,((((2.7e-2f*xdn(1,lc))*cx(1,lc))*xv(1,lc))*(((((0.5e20f*lake_pow(rb(1),3))*xdia(1,lc,1)))-0.4f))));
// WRF:15199
if((rb(1)>3.51e-6f)) {
// WRF:15201
rh(1) = lake_max(41.e-6,lake_div(6.3e-4,((1.e6*((rb(1)-3.5e-6))))));
// WRF:15202
} else {
// WRF:15203
rh(1) = 41.e-6;
// WRF:15204
}
// WRF:15205
if((xl2p(1)>0.0f)) {
// WRF:15206
nh(1) = (4.2e9*xl2p(1));
// WRF:15207
} else {
// WRF:15208
nh(1) = 1.e30f;
// WRF:15209
}
// WRF:15210
}
// WRF:15211
}
// WRF:15220
if((ndebug>=0)) {
// WRF:15222
}
// WRF:15223
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15224
qvimxd(1) = ((0.70f*((qx(1,lv)-qis(1))))*dtpinv);
// WRF:15226
if((qx(1,lc)<coeff_qxmin{}(lc))) {qvimxd(1) = ((0.99f*((qx(1,lv)-qis(1))))*dtpinv);}
// WRF:15228
qvimxd(1) = lake_max(qvimxd(1),0.0f);
// WRF:15230
frac = 0.1e0;
// WRF:15231
qimxd(1) = ((frac*qx(1,li))*dtpinv);
// WRF:15232
qcmxd(1) = ((frac*qx(1,lc))*dtpinv);
// WRF:15233
qrmxd(1) = ((frac*qx(1,lr))*dtpinv);
// WRF:15234
qsmxd(1) = ((frac*qx(1,ls))*dtpinv);
// WRF:15235
qhmxd(1) = ((frac*qx(1,lh))*dtpinv);
// WRF:15236
if((lhl>1)) {qhlmxd(1) = ((frac*qx(1,lhl))*dtpinv);}
// WRF:15237
}
// WRF:15239
if((ndebug>=0)) {
// WRF:15241
}
// WRF:15243
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15245
if((qx(1,lc)<=coeff_qxmin{}(lc))) {
// WRF:15246
ccmxd(1) = ((0.20f*cx(1,lc))*dtpinv);
// WRF:15247
} else {
// WRF:15248
if((ipconc>=2)) {
// WRF:15249
ccmxd(1) = ((frac*cx(1,lc))*dtpinv);
// WRF:15250
} else {
// WRF:15251
ccmxd(1) = lake_div((frac*qx(1,lc)),(((xmas(1,lc)*rho0(1))*dtp)));
// WRF:15252
}
// WRF:15253
}
// WRF:15255
if((qx(1,li)<=coeff_qxmin{}(li))) {
// WRF:15256
cimxd(1) = ((frac*cx(1,li))*dtpinv);
// WRF:15257
} else {
// WRF:15258
if((ipconc>=1)) {
// WRF:15259
cimxd(1) = ((frac*cx(1,li))*dtpinv);
// WRF:15260
} else {
// WRF:15261
cimxd(1) = lake_div((frac*qx(1,li)),(((xmas(1,li)*rho0(1))*dtp)));
// WRF:15262
}
// WRF:15263
}
// WRF:15266
crmxd(1) = ((0.10f*cx(1,lr))*dtpinv);
// WRF:15267
csmxd(1) = ((frac*cx(1,ls))*dtpinv);
// WRF:15268
chmxd(1) = ((frac*cx(1,lh))*dtpinv);
// WRF:15270
ccmxd(1) = ((frac*cx(1,lc))*dtpinv);
// WRF:15271
cimxd(1) = ((frac*cx(1,li))*dtpinv);
// WRF:15272
crmxd(1) = ((frac*cx(1,lr))*dtpinv);
// WRF:15273
csmxd(1) = ((frac*cx(1,ls))*dtpinv);
// WRF:15274
chmxd(1) = ((frac*cx(1,lh))*dtpinv);
// WRF:15276
qxmxd(1,lv) = lake_max(0.0f,((0.1f*((qx(1,lv)-qvs(1))))*dtpinv));
// WRF:15278
#pragma unroll
for(il=lc;il<=lhab;il+=1) {
// WRF:15279
qxmxd(1,il) = ((frac*qx(1,il))*dtpinv);
// WRF:15280
cxmxd(1,il) = ((frac*cx(1,il))*dtpinv);
// WRF:15281
}
// WRF:15283
}
// WRF:15288
if((ipconc>=6)) {
// WRF:15289
frac = 0.4e0;
// WRF:15290
for(int _s0=1;_s0<=ngs;++_s0) {for(int _s1=lr;_s1<=lhab;++_s1) {zxmxd(_s0,_s1) = 0.0f;}}
// WRF:15291
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:15292
if(((coeff_lz{}(il)>0)||((il==lr)))) {
// WRF:15293
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15294
zxmxd(1,il) = ((frac*zx(1,il))*dtpinv);
// WRF:15295
}
// WRF:15296
}
// WRF:15297
}
// WRF:15298
}
// WRF:15304
maxmassfac(lc) = (lake_div(lake_pow(((2.f+(3.f*((1.f+coeff_xnu{}(lc)))))),3),((3.f*((1.f+coeff_xnu{}(lc)))))));
// WRF:15305
maxmassfac(li) = (lake_div(lake_pow(((2.f+(3.f*((1.f+coeff_xnu{}(li)))))),3),((3.f*((1.f+coeff_xnu{}(li)))))));
// WRF:15307
if((imurain==3)) {
// WRF:15308
maxmassfac(lr) = (lake_div(lake_pow(((2.f+(3.f*((1.f+coeff_xnu{}(lr)))))),3),((3.f*((1.f+coeff_xnu{}(lr)))))));
// WRF:15309
} else {
// WRF:15310
maxmassfac(lr) = lake_div(lake_pow(((3.0f+alphar)),3),(((((3.f+alphar))*((2.f+alphar)))*((1.f+alphar)))));
// WRF:15312
}
// WRF:15314
if((imusnow==3)) {
// WRF:15315
maxmassfac(ls) = (lake_div(lake_pow(((2.f+(3.f*((1.f+alphas))))),3),((3.f*((1.f+alphas))))));
// WRF:15316
} else {
// WRF:15317
maxmassfac(ls) = lake_div(lake_pow(((3.0f+alphas)),3),(((((3.f+alphas))*((2.f+alphas)))*((1.f+alphas)))));
// WRF:15319
}
// WRF:15321
maxmassfac(lh) = lake_div(lake_pow(((3.0f+alphah)),3),(((((3.f+alphah))*((2.f+alphah)))*((1.f+alphah)))));
// WRF:15324
if((lhl>1)) {
// WRF:15325
maxmassfac(lhl) = lake_div(lake_pow(((3.0f+alphahl)),3),(((((3.f+alphahl))*((2.f+alphahl)))*((1.f+alphahl)))));
// WRF:15327
}
// WRF:15331
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15332
#pragma unroll
for(il=lh;il<=lhab;il+=1) {
// WRF:15334
vshdgs(1,il) = vshd;
// WRF:15336
if(((qx(1,il)>coeff_qxmin{}(il))&&(ivshdgs>0))) {
// WRF:15339
tmpdiam = (((shedalp+alpha(1,il)))*xdia(1,il,1));
// WRF:15341
if((tmpdiam>sheddiam0)) {
// WRF:15342
vshdgs(1,il) = lake_div((0.523599f*lake_pow((1.5e-3f),3)),massfacshr);
// WRF:15343
} else if((tmpdiam>sheddiam)) {
// WRF:15344
vshdgs(1,il) = lake_div((0.523599f*lake_pow((3.0e-3f),3)),massfacshr);
// WRF:15345
} else {
// WRF:15347
vshdgs(1,il) = lake_div(lake_min(coeff_xvmx{}(lr),(((lake_div(6.f,pi)*xdn(1,il))*0.001f)*lake_pow(tmpdiam,3))),massfacshr);
// WRF:15348
}
// WRF:15349
}
// WRF:15350
}
// WRF:15351
}
// WRF:15361
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:15363
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15367
qcwresv(1) = 0.0f;
// WRF:15368
ccwresv(1) = 0.0f;
// WRF:15370
erw(1) = 0.0f;
// WRF:15371
esw(1) = 0.0f;
// WRF:15372
ehw(1) = 0.0f;
// WRF:15373
efw(1) = 0.0f;
// WRF:15374
ehlw(1) = 0.0f;
// WRF:15377
err(1) = 0.0f;
// WRF:15378
esr(1) = 0.0f;
// WRF:15379
il2(1) = 0;
// WRF:15380
il3(1) = 0;
// WRF:15381
ehr(1) = 0.0f;
// WRF:15382
ehlr(1) = 0.0f;
// WRF:15385
eri(1) = 0.0f;
// WRF:15386
esi(1) = 0.0f;
// WRF:15387
ehi(1) = 0.0f;
// WRF:15388
ehis(1) = 0.0f;
// WRF:15389
ehli(1) = 0.0f;
// WRF:15390
ehlis(1) = 0.0f;
// WRF:15393
ers(1) = 0.0f;
// WRF:15394
ess(1) = 0.0f;
// WRF:15395
ehs(1) = 0.0f;
// WRF:15396
ehsfac(1) = 1.0f;
// WRF:15397
ehls(1) = 0.0f;
// WRF:15398
ehscnv(1) = 0.0f;
// WRF:15401
eiw(1) = 0.0f;
// WRF:15402
eii(1) = 0.0f;
// WRF:15403
ehsclsn(1) = 0.0f;
// WRF:15404
ehiclsn(1) = 0.0f;
// WRF:15405
ehlsclsn(1) = 0.0f;
// WRF:15406
ehliclsn(1) = 0.0f;
// WRF:15407
esiclsn(1) = 0.0f;
// WRF:15411
if(((exwmindiam>0)&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15412
tmp = (cx(1,lc)*gfk_exp((-lake_pow((lake_div(exwmindiam,xdia(1,lc,1))),3))));
// WRF:15413
ccwresv(1) = lake_min(cx(1,lc),lake_max(2.e6f,(cx(1,lc)-tmp)));
// WRF:15415
tmp = (cx(1,lc)-ccwresv(1));
// WRF:15417
volt = (lake_div(pi,6.f)*lake_pow((exwmindiam),3));
// WRF:15418
qcwresv(1) = (qx(1,lc)-(((tmp*coeff_xdn0{}(lc))*rhoinv(1))*((volt+xv(1,lc)))));
// WRF:15421
if((false&&(qx(1,lc)>0.1e-3f))) {
// WRF:15423
; /* native diagnostic only */
// WRF:15425
}
// WRF:15427
}
// WRF:15430
icwr(1) = 1;
// WRF:15431
if((qx(1,lc)>coeff_qxmin{}(lc))) {
// WRF:15432
cwrad = (0.5f*xdia(1,lc,1));
// WRF:15433
#pragma unroll
for(il=1;il<=8;il+=1) {
// WRF:15434
if((cwrad>=(1.e-6f*cwr(il,1)))) {icwr(1) = il;}
// WRF:15435
}
// WRF:15436
}
// WRF:15439
irwr(1) = 1;
// WRF:15440
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:15441
rwrad = (0.5f*xdia(1,lr,3));
// WRF:15442
#pragma unroll
for(il=1;il<=6;il+=1) {
// WRF:15443
if((rwrad>=(1.e-6f*grad(il,1)))) {irwr(1) = il;}
// WRF:15444
}
// WRF:15445
}
// WRF:15448
igwr(1) = 1;
// WRF:15452
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:15453
rwrad = (0.5f*xdia(1,lh,3));
// WRF:15454
#pragma unroll
for(il=1;il<=6;il+=1) {
// WRF:15455
if((rwrad>=(1.e-6f*grad(il,1)))) {igwr(1) = il;}
// WRF:15456
}
// WRF:15457
}
// WRF:15460
if((lhl>1)) {
// WRF:15461
ihlr(1) = 1;
// WRF:15462
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:15463
rwrad = (0.5f*xdia(1,lhl,3));
// WRF:15464
#pragma unroll
for(il=1;il<=6;il+=1) {
// WRF:15465
if((rwrad>=(1.e-6f*grad(il,1)))) {ihlr(1) = il;}
// WRF:15466
}
// WRF:15467
}
// WRF:15468
}
// WRF:15475
if((qx(1,li)>coeff_qxmin{}(li))) {
// WRF:15483
eii(1) = gfk_exp((0.025f*lake_min(temcg(1),0.0f)));
// WRF:15485
if((temg(1)>273.15f)) {eii(1) = 1.0f;}
// WRF:15486
}
// WRF:15493
eiw(1) = 0.0f;
// WRF:15494
if(((qx(1,li)>coeff_qxmin{}(li))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15497
if(((xdia(1,lc,1)>ewi_dcmin)&&(xdia(1,li,1)>ewi_dimin))) {
// WRF:15500
eiw(1) = 0.5f;
// WRF:15501
}
// WRF:15502
if((temg(1)>=273.15f)) {eiw(1) = 0.0f;}
// WRF:15503
}
// WRF:15511
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15513
if((lnr>1)) {
// WRF:15514
erw(1) = 1.0f;
// WRF:15516
} else {
// WRF:15527
ic = icwr(1);
// WRF:15528
icp1 = lake_min(8,(ic+1));
// WRF:15529
ir = irwr(1);
// WRF:15530
irp1 = lake_min(6,(ir+1));
// WRF:15531
cwrad = (0.5f*xdia(1,lc,3));
// WRF:15532
rwrad = (0.5f*xdia(1,lr,3));
// WRF:15534
slope1 = (((ew(icp1,ir)-ew(ic,ir)))*cwr(ic,2));
// WRF:15535
slope2 = (((ew(icp1,irp1)-ew(ic,irp1)))*cwr(ic,2));
// WRF:15539
x1 = (ew(ic,ir)+(slope1*lake_max(0.0f,((cwrad-cwr(ic,1))))));
// WRF:15540
x2 = (ew(icp1,ir)+(slope2*lake_max(0.0f,((cwrad-cwr(ic,1))))));
// WRF:15542
slope1 = (((x2-x1))*grad(ir,2));
// WRF:15544
erw(1) = lake_max(0.0f,(x1+(slope1*lake_max(0.0f,((rwrad-grad(ir,1)))))));
// WRF:15549
erw(1) = lake_max(0.0f,erw(1));
// WRF:15550
if((rwrad<50.e-6f)) {
// WRF:15551
erw(1) = 0.0f;
// WRF:15552
} else if((rwrad<100.e-6f)) {
// WRF:15553
erw(1) = lake_div((erw(1)*((rwrad-50.e-6f))),50.e-6f);
// WRF:15554
}
// WRF:15556
}
// WRF:15557
}
// WRF:15558
if((cx(1,lc)<=0.0f)) {erw(1) = 0.0f;}
// WRF:15560
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:15561
err(1) = 1.0f;
// WRF:15562
}
// WRF:15564
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(qx(1,ls)>coeff_qxmin{}(ls)))) {
// WRF:15565
ers(1) = 1.0f;
// WRF:15566
}
// WRF:15568
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:15571
eri(1) = eri0;
// WRF:15578
if((xdia(1,li,3)<eri_cimin)) {eri(1) = 0.0f;}
// WRF:15579
}
// WRF:15589
esw(1) = 0.0f;
// WRF:15590
if(((qx(1,ls)>coeff_qxmin{}(ls))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15591
esw(1) = 0.5f;
// WRF:15592
if(((xdia(1,lc,1)>15.e-6f)&&(xdia(1,ls,1)>100.e-6f))) {
// WRF:15593
esw(1) = 0.5f;
// WRF:15594
} else if((xdia(1,ls,1)>=500.e-6f)) {
// WRF:15595
esw(1) = lake_min(0.5f,(0.05f+(lake_div(((0.8f-0.05f)),(40.e-6f))*xdia(1,lc,1))));
// WRF:15596
}
// WRF:15597
}
// WRF:15599
if((((qx(1,ls)>coeff_qxmin{}(ls))&&(qx(1,lr)>coeff_qxmin{}(lr)))&&(temg(1)<(tfr-1.f)))) {
// WRF:15602
esr(1) = (gfk_exp(lake_div((-lake_pow((40.e-6f),3)),xv(1,lr)))*gfk_exp(lake_div((-40.e-6f),xdia(1,ls,1))));
// WRF:15603
if(((qx(1,ls)<1.e-4f)&&(qx(1,lr)<1.e-4f))) {il2(1) = 1;}
// WRF:15604
}
// WRF:15606
if(((((ipconc<3)&&(temg(1)<tfr))&&(qx(1,lr)>coeff_qxmin{}(lr)))&&(qx(1,lr)<1.e-4f))) {
// WRF:15607
il3(1) = 1;
// WRF:15608
}
// WRF:15611
if((temcg(1)<0.0f)) {
// WRF:15613
if(((ipconc<4)||(temcg(1)<esstem1))) {
// WRF:15614
ess(1) = 0.0f;
// WRF:15618
} else {
// WRF:15620
fac = lake_abs(ess0);
// WRF:15621
if((iessopt==2)) {
// WRF:15623
if((wvel(1)>2.0f)) {
// WRF:15625
fac = 0.0f;
// WRF:15626
} else if((wvel(1)>1.0f)) {
// WRF:15627
fac = (lake_max(0.0f,(2.0f-wvel(1)))*fac);
// WRF:15628
}
// WRF:15629
} else if((iessopt==3)) {
// WRF:15630
if((ssi(1)<=1.0f)) {
// WRF:15631
fac = 0.0f;
// WRF:15632
ehsfac(1) = 0.0f;
// WRF:15633
} else if((ssi(1)<=1.02f)) {
// WRF:15634
fac = lake_div((fac*((ssi(1)-1.0f))),0.02f);
// WRF:15635
ehsfac(1) = lake_div(((ssi(1)-1.0f)),0.02f);
// WRF:15636
}
// WRF:15637
} else if((iessopt==4)) {
// WRF:15638
if((ssi(1)<=1.0f)) {
// WRF:15639
fac = 0.1f;
// WRF:15640
ehsfac(1) = 0.1f;
// WRF:15641
} else if((ssi(1)<=1.005f)) {
// WRF:15642
fac = lake_max(0.1f,lake_div((fac*((ssi(1)-1.0f))),0.005f));
// WRF:15643
ehsfac(1) = lake_max(0.1f,lake_div(((ssi(1)-1.0f)),0.005f));
// WRF:15644
}
// WRF:15645
}
// WRF:15647
if(((temcg(1)>esstem1)&&(temcg(1)<esstem2))) {
// WRF:15648
ess(1) = lake_div(((fac*gfk_exp((ess1*(esstem2))))*((temcg(1)-esstem1))),((esstem2-esstem1)));
// WRF:15649
} else if((temcg(1)>=esstem2)) {
// WRF:15650
ess(1) = (fac*gfk_exp((ess1*lake_min(temcg(1),0.0f))));
// WRF:15651
}
// WRF:15653
}
// WRF:15654
}
// WRF:15656
if(((qx(1,ls)>coeff_qxmin{}(ls))&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:15657
esiclsn(1) = esi_collsn;
// WRF:15659
if(((ipconc<1)&&lwsm6)) {
// WRF:15660
esi(1) = gfk_exp((0.7f*lake_min(temcg(1),0.0f)));
// WRF:15661
} else {
// WRF:15662
esi(1) = (esi0*gfk_exp((0.1f*lake_min(temcg(1),0.0f))));
// WRF:15663
esi(1) = lake_min(0.1f,esi(1));
// WRF:15664
}
// WRF:15665
if((ipconc<=3)) {
// WRF:15666
esi(1) = gfk_exp((0.025f*lake_min(temcg(1),0.0f)));
// WRF:15669
}
// WRF:15673
if((temg(1)>273.15f)) {esi(1) = 0.0f;}
// WRF:15674
}
// WRF:15682
xmascw(1) = xmas(1,lc);
// WRF:15683
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15684
ehw(1) = 1.0f;
// WRF:15685
if((iehw==0)) {
// WRF:15686
ehw(1) = ehw0;
// WRF:15687
} else if(((iehw==1)||(iehw==10))) {
// WRF:15688
cwrad = (0.5f*xdia(1,lc,1));
// WRF:15689
ehw(1) = lake_min(ehw0,(ewfac*lake_min(((aradcw+(cwrad*((bradcw+(cwrad*((cradcw+(cwrad*(dradcw)))))))))),1.0f)));
// WRF:15693
} else if(((iehw==2)||(iehw==10))) {
// WRF:15694
ic = icwr(1);
// WRF:15695
icp1 = lake_min(8,(ic+1));
// WRF:15696
ir = igwr(1);
// WRF:15697
irp1 = lake_min(6,(ir+1));
// WRF:15698
cwrad = (0.5f*xdia(1,lc,1));
// WRF:15699
rwrad = (0.5f*xdia(1,lh,3));
// WRF:15701
slope1 = (((ew(icp1,ir)-ew(ic,ir)))*cwr(ic,2));
// WRF:15702
slope2 = (((ew(icp1,irp1)-ew(ic,irp1)))*cwr(ic,2));
// WRF:15706
x1 = (ew(ic,ir)+(slope1*lake_max(0.0f,((cwrad-cwr(ic,1))))));
// WRF:15707
x2 = (ew(icp1,ir)+(slope2*lake_max(0.0f,((cwrad-cwr(ic,1))))));
// WRF:15709
slope1 = (((x2-x1))*grad(ir,2));
// WRF:15711
tmp = lake_max(0.0f,lake_min(1.0f,(x1+(slope1*lake_max(0.0f,((rwrad-grad(ir,1))))))));
// WRF:15712
ehw(1) = lake_min(ehw(1),tmp);
// WRF:15722
} else if(((iehw==3)||(iehw==10))) {
// WRF:15723
tmp = gfk_exp((-lake_pow((lake_div(dmincw,xdia(1,lc,1))),3)));
// WRF:15724
xmascw(1) = (xmas(1,lc)+(coeff_xdn0{}(lc)*(lake_div((pi*lake_pow(dmincw,3)),6.0f))));
// WRF:15725
ehw(1) = lake_min(ehw(1),tmp);
// WRF:15726
} else if(((iehw==4)||(iehw==10))) {
// WRF:15727
tmp = lake_div((((2.0f*xdn(1,lc))*vtxbar(1,lh,1))*lake_pow(((0.5f*xdia(1,lc,1))),2)),((((9.0f*fadvisc(1))*0.5f)*xdia(1,lh,3))));
// WRF:15730
tmp = lake_max(1.5f,lake_min(10.0f,tmp));
// WRF:15731
ehw(1) = lake_min(ehw(1),(0.55f*nssl1_log10f((2.51f*tmp))));
// WRF:15732
}
// WRF:15733
if((xdia(1,lc,1)<2.4e-06f)) {ehw(1) = 0.0f;}
// WRF:15735
ehw(1) = lake_min(ehw0,ehw(1));
// WRF:15737
if(((ibfc==(-1))&&(temcg(1)<(-41.0f)))) {
// WRF:15738
ehw(1) = 0.0f;
// WRF:15739
}
// WRF:15741
}
// WRF:15743
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:15748
ehr(1) = (gfk_exp(lake_div((-(40.e-6f)),xdia(1,lr,3)))*gfk_exp(lake_div((-40.e-6f),xdia(1,lh,3))));
// WRF:15749
ehr(1) = lake_min(ehr0,ehr(1));
// WRF:15750
}
// WRF:15752
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:15753
if((ipconc>=4)) {
// WRF:15754
ehscnv(1) = (ehs0*gfk_exp((ehs1*lake_min(temcg(1),0.0f))));
// WRF:15755
} else {
// WRF:15756
ehscnv(1) = gfk_exp((0.09f*lake_min(temcg(1),0.0f)));
// WRF:15757
}
// WRF:15759
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,lc)>=coeff_qxmin{}(lc)))) {
// WRF:15763
ehsclsn(1) = ehs_collsn;
// WRF:15764
if((xdia(1,ls,3)<40.e-6f)) {
// WRF:15765
ehsclsn(1) = 0.0f;
// WRF:15766
} else if((xdia(1,ls,3)<150.e-6f)) {
// WRF:15767
ehsclsn(1) = lake_div((ehs_collsn*((xdia(1,ls,3)-40.e-6f))),((150.e-6f-40.e-6f)));
// WRF:15768
} else {
// WRF:15769
ehsclsn(1) = ehs_collsn;
// WRF:15770
}
// WRF:15772
ehs(1) = (ehscnv(1)*lake_min(1.0f,lake_div(lake_max(0.0f,(xdn(1,lh)-300.f)),300.f)));
// WRF:15774
ehs(1) = lake_min(ehs(1),ehsmax);
// WRF:15775
}
// WRF:15776
}
// WRF:15778
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:15779
ehiclsn(1) = ehi_collsn;
// WRF:15780
ehi(1) = (eii0*gfk_exp((eii1*lake_min(temcg(1),0.0f))));
// WRF:15781
ehi(1) = lake_min(ehimax,lake_max(ehi(1),ehimin));
// WRF:15783
}
// WRF:15785
if((lis>1)) {
// WRF:15786
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,lis)>coeff_qxmin{}(lis)))) {
// WRF:15787
ehisclsn(1) = ehi_collsn;
// WRF:15788
ehis(1) = (eii0*gfk_exp((eii1*lake_min(temcg(1),0.0f))));
// WRF:15789
ehis(1) = lake_min(ehimax,lake_max(ehis(1),ehimin));
// WRF:15791
}
// WRF:15792
}
// WRF:15800
if((lhl>1)) {
// WRF:15802
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:15803
if((iehw==3)) {iehlw = 3;}
// WRF:15804
if((iehw==4)) {iehlw = 4;}
// WRF:15805
ehlw(1) = ehlw0;
// WRF:15806
if((iehlw==0)) {
// WRF:15807
ehlw(1) = ehlw0;
// WRF:15808
} else if(((iehlw==1)||(iehlw==10))) {
// WRF:15809
cwrad = (0.5f*xdia(1,lc,1));
// WRF:15810
ehlw(1) = lake_min(ehlw0,(ewfac*lake_min(((aradcw+(cwrad*((bradcw+(cwrad*((cradcw+(cwrad*(dradcw)))))))))),1.0f)));
// WRF:15814
} else if(((iehlw==2)||(iehlw==10))) {
// WRF:15815
ic = icwr(1);
// WRF:15816
icp1 = lake_min(8,(ic+1));
// WRF:15817
ir = ihlr(1);
// WRF:15818
irp1 = lake_min(6,(ir+1));
// WRF:15819
cwrad = (0.5f*xdia(1,lc,1));
// WRF:15820
rwrad = (0.5f*xdia(1,lhl,3));
// WRF:15822
slope1 = (((ew(icp1,ir)-ew(ic,ir)))*cwr(ic,2));
// WRF:15823
slope2 = (((ew(icp1,irp1)-ew(ic,irp1)))*cwr(ic,2));
// WRF:15825
x1 = (ew(ic,ir)+(slope1*((cwrad-cwr(ic,1)))));
// WRF:15826
x2 = (ew(icp1,ir)+(slope2*((cwrad-cwr(ic,1)))));
// WRF:15828
slope1 = (((x2-x1))*grad(ir,2));
// WRF:15830
tmp = lake_max(0.0f,lake_min(1.0f,(x1+(slope1*((rwrad-grad(ir,1)))))));
// WRF:15831
ehlw(1) = lake_min(ehlw(1),tmp);
// WRF:15832
ehlw(1) = lake_min(ehlw0,ehlw(1));
// WRF:15838
} else if(((iehlw==3)||(iehlw==10))) {
// WRF:15839
tmp = gfk_exp((-lake_pow((lake_div(dmincw,xdia(1,lc,1))),3)));
// WRF:15840
ehlw(1) = lake_min(ehlw(1),tmp);
// WRF:15841
} else if(((iehlw==4)||(iehlw==10))) {
// WRF:15842
tmp = lake_div((((2.0f*xdn(1,lc))*vtxbar(1,lhl,1))*lake_pow(((0.5f*xdia(1,lc,1))),2)),((((9.0f*fadvisc(1))*0.5f)*xdia(1,lhl,3))));
// WRF:15845
tmp = lake_max(1.5f,lake_min(10.0f,tmp));
// WRF:15846
ehlw(1) = lake_min(ehlw(1),(0.55f*nssl1_log10f((2.51f*tmp))));
// WRF:15847
}
// WRF:15848
if((xdia(1,lc,1)<2.4e-06f)) {ehlw(1) = 0.0f;}
// WRF:15849
ehlw(1) = lake_min(ehlw0,ehlw(1));
// WRF:15851
if(((ibfc==(-1))&&(temcg(1)<(-41.0f)))) {
// WRF:15852
ehlw(1) = 0.0f;
// WRF:15853
}
// WRF:15855
}
// WRF:15857
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:15860
ehlr(1) = 1.0f;
// WRF:15861
ehlr(1) = lake_min(ehlr0,ehlr(1));
// WRF:15862
}
// WRF:15864
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:15865
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:15866
ehlsclsn(1) = ehls_collsn;
// WRF:15867
ehls(1) = ehscnv(1);
// WRF:15868
ehls(1) = lake_min(ehls(1),ehsmax);
// WRF:15869
}
// WRF:15870
}
// WRF:15872
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:15873
ehliclsn(1) = ehli_collsn;
// WRF:15874
ehli(1) = (eii0hl*gfk_exp((eii1hl*lake_min(temcg(1),0.0f))));
// WRF:15875
ehli(1) = lake_min(ehimax,lake_max(ehli(1),ehimin));
// WRF:15876
if(((temg(1)>273.15f)||((qx(1,lc)<coeff_qxmin{}(lc))))) {ehli(1) = 0.0f;}
// WRF:15877
}
// WRF:15879
if((lis>1)) {
// WRF:15880
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(qx(1,lis)>coeff_qxmin{}(lis)))) {
// WRF:15881
ehlisclsn(1) = ehli_collsn;
// WRF:15882
ehlis(1) = (eii0*gfk_exp((eii1*lake_min(temcg(1),0.0f))));
// WRF:15883
ehlis(1) = lake_min(ehimax,lake_max(ehlis(1),ehimin));
// WRF:15884
if(((temg(1)>273.15f)||((qx(1,lc)<coeff_qxmin{}(lc))))) {ehlis(1) = 0.0f;}
// WRF:15885
}
// WRF:15886
}
// WRF:15889
}
// WRF:15891
}
// WRF:15899
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15901
xplate(1) = 0.0f;
// WRF:15902
xcolmn(1) = 1.0f;
// WRF:15924
}
// WRF:15934
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:15936
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15937
qracw(1) = 0.0f;
// WRF:15938
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(erw(1)>0.0f))) {
// WRF:15939
if((ipconc<3)) {
// WRF:15940
if(((erw(1)>0.0f)&&(qx(1,lr)>1.e-7f))) {
// WRF:15941
vt = (((ar*(lake_pow(xdia(1,lc,1),br))))*rhovt(1));
// WRF:15942
qracw(1) = (((((((0.25f)*pi)*erw(1))*((qx(1,lc)-qcwresv(1))))*cx(1,lr))*lake_max(0.0f,(vtxbar(1,lr,1)-vt)))*((((gf3*xdia(1,lr,2))+(((2.0f*gf2)*xdia(1,lr,1))*xdia(1,lc,1)))+(gf1*xdia(1,lc,2)))));
// WRF:15954
}
// WRF:15955
} else {
// WRF:15957
if(((dmrauto<=0)||((rho0(1)*qx(1,lr))>(1.2f*xl2p(1))))) {
// WRF:15958
rwrad = (0.5f*xdia(1,lr,3));
// WRF:15959
if((rwrad>rh(1))) {
// WRF:15960
if((rwrad>rwradmn)) {
// WRF:15963
qracw(1) = lake_div((((((erw(1)*aa2)*cx(1,lr))*cx(1,lc))*xmas(1,lc))*((lake_div((((alpha(1,lc)+2.f))*xv(1,lc)),((alpha(1,lc)+1.f)))+xv(1,lr)))),rho0(1));
// WRF:15965
} else {
// WRF:15967
if((imurain==3)) {
// WRF:15976
qracw(1) = (((aa1*cx(1,lr))*((qx(1,lc)-qcwresv(1))))*((lake_div(((((alpha(1,lc)+3.f))*((alpha(1,lc)+2.f)))*lake_pow(xv(1,lc),2)),lake_pow(((alpha(1,lc)+1.f)),2))+lake_div((((alpha(1,lr)+2.f))*lake_pow(xv(1,lr),2)),((alpha(1,lr)+1.f))))));
// WRF:15980
} else {
// WRF:15982
qracw(1) = (((aa1*cx(1,lr))*((qx(1,lc)-qcwresv(1))))*((lake_div(((((alpha(1,lc)+3.f))*((alpha(1,lc)+2.f)))*lake_pow(xv(1,lc),2)),lake_pow(((alpha(1,lc)+1.f)),2))+lake_div((((((alpha(1,lr)+6.f))*((alpha(1,lr)+5.f)))*((alpha(1,lr)+4.f)))*lake_pow(xv(1,lr),2)),(((((alpha(1,lr)+3.f))*((alpha(1,lr)+2.f)))*((alpha(1,lr)+1.f))))))));
// WRF:15987
}
// WRF:15989
}
// WRF:15990
}
// WRF:15991
}
// WRF:15992
}
// WRF:15994
qracw(1) = lake_min(qracw(1),qcmxd(1));
// WRF:15995
}
// WRF:15996
}
// WRF:15998
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:15999
qraci(1) = 0.0f;
// WRF:16000
craci(1) = 0.0f;
// WRF:16001
qracs(1) = 0.0f;
// WRF:16002
if((((eri(1)>0.0f)&&(iacr>=1))&&(xdia(1,lr,3)>(2.f*rwradmn)))) {
// WRF:16003
if((ipconc>=3)) {
// WRF:16005
tmp = ((((eri(1)*aa2)*cx(1,lr))*cx(1,li))*((lake_div((((cinu+2.f))*xv(1,li)),((cinu+1.f)))+xv(1,lr))));
// WRF:16008
qraci(1) = lake_min(qxmxd(1,li),((tmp*xmas(1,li))*rhoinv(1)));
// WRF:16009
craci(1) = lake_min(cxmxd(1,li),tmp);
// WRF:16031
} else {
// WRF:16032
qraci(1) = lake_min((((((((0.25f)*pi)*eri(1))*qx(1,li))*cx(1,lr))*lake_abs((vtxbar(1,lr,1)-vtxbar(1,li,1))))*((((gf3*xdia(1,lr,2))+(((2.0f*gf2)*xdia(1,lr,1))*xdia(1,li,1)))+(gf1*xdia(1,li,2))))),qimxd(1));
// WRF:16040
}
// WRF:16041
if((temg(1)>268.15f)) {
// WRF:16042
qraci(1) = 0.0f;
// WRF:16043
}
// WRF:16044
}
// WRF:16045
}
// WRF:16047
if((ipconc<3)) {
// WRF:16048
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16049
qracs(1) = 0.0f;
// WRF:16050
if(((ers(1)>0.0f)&&(ipconc<3))) {
// WRF:16051
if((lwsm6&&(ipconc==0))) {
// WRF:16052
vt = vt2ave(1);
// WRF:16053
} else {
// WRF:16054
vt = vtxbar(1,ls,1);
// WRF:16055
}
// WRF:16056
qracs(1) = lake_min(((((((lake_div(((0.25f)*pi),gf4))*ers(1))*qx(1,ls))*cx(1,lr))*lake_abs((vtxbar(1,lr,1)-vt)))*(((((gf6*gf1)*xdia(1,ls,2))+((((2.0f*gf5)*gf2)*xdia(1,ls,1))*xdia(1,lr,1)))+((gf4*gf3)*xdia(1,lr,2))))),qsmxd(1));
// WRF:16064
}
// WRF:16065
}
// WRF:16066
}
// WRF:16070
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16072
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16073
qsacw(1) = 0.0f;
// WRF:16074
csacw(1) = 0.0f;
// WRF:16075
vsacw(1) = 0.0f;
// WRF:16076
if((esw(1)>0.0f)) {
// WRF:16078
if((ipconc>=4)) {
// WRF:16084
tmp = (((((1.0f*rvt)*aa2)*cx(1,ls))*cx(1,lc))*((lake_div((((alpha(1,lc)+2.f))*xv(1,lc)),((alpha(1,lc)+1.f)))+xv(1,ls))));
// WRF:16087
qsacw(1) = lake_min(qxmxd(1,lc),((tmp*xmas(1,lc))*rhoinv(1)));
// WRF:16088
csacw(1) = lake_min(cxmxd(1,lc),tmp);
// WRF:16090
if((coeff_lvol{}(ls)>1)) {
// WRF:16091
if((temg(1)<273.15f)) {
// WRF:16092
rimdn(1,ls) = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*vtxbar(1,ls,1)))),((temg(1)-273.15f)))),(rimc2)));
// WRF:16095
rimdn(1,ls) = lake_min(lake_max(rimc3,rimdn(1,ls)),rimc4);
// WRF:16096
} else {
// WRF:16097
rimdn(1,ls) = 1000.f;
// WRF:16098
}
// WRF:16100
vsacw(1) = lake_div((rho0(1)*qsacw(1)),rimdn(1,ls));
// WRF:16102
}
// WRF:16107
} else {
// WRF:16117
vt = lake_abs((vtxbar(1,ls,1)-vtxbar(1,lc,1)));
// WRF:16119
qsacw(1) = ((((((0.25f*pi)*esw(1))*cx(1,ls))*qx(1,lc))*vt)*((((coeff_da0{}(ls)*lake_pow(xdia(1,ls,3),2))+((coeff_dab1{}(ls,lc)*xdia(1,ls,3))*xdia(1,lc,3)))+(da1lc(1)*lake_pow(xdia(1,lc,3),2)))));
// WRF:16123
qsacw(1) = lake_min(qsacw(1),qxmxd(1,ls));
// WRF:16124
csacw(1) = lake_div((rho0(1)*qsacw(1)),xmas(1,lc));
// WRF:16125
}
// WRF:16126
}
// WRF:16127
}
// WRF:16130
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16131
qsaci(1) = 0.0f;
// WRF:16132
csaci(1) = 0.0f;
// WRF:16133
csaci0(1) = 0.0f;
// WRF:16134
if((ipconc>=4)) {
// WRF:16135
if(((esi(1)>0.0f)||(((ipelec>0)&&(esiclsn(1)>0.0f))))) {
// WRF:16139
tmp = (((((esiclsn(1)*rvt)*aa2)*cx(1,ls))*cx(1,li))*((lake_div((((cinu+2.f))*xv(1,li)),((cinu+1.f)))+xv(1,ls))));
// WRF:16142
qsaci(1) = lake_min(qxmxd(1,li),(((esi(1)*tmp)*xmas(1,li))*rhoinv(1)));
// WRF:16143
csaci0(1) = tmp;
// WRF:16144
csaci(1) = lake_min(cxmxd(1,li),(esi(1)*tmp));
// WRF:16154
}
// WRF:16155
} else {
// WRF:16156
if((esi(1)>0.0f)) {
// WRF:16157
qsaci(1) = lake_min(((((((((0.25f)*pi))*esi(1))*qx(1,li))*cx(1,ls))*lake_abs((vtxbar(1,ls,1)-vtxbar(1,li,1))))*((((gf3*xdia(1,ls,2))+(((2.0f*gf2)*xdia(1,ls,1))*xdia(1,li,1)))+(gf1*xdia(1,li,2))))),qimxd(1));
// WRF:16165
}
// WRF:16166
}
// WRF:16167
}
// WRF:16171
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16172
qsacr(1) = 0.0f;
// WRF:16173
qsacrs(1) = 0.0f;
// WRF:16174
csacr(1) = 0.0f;
// WRF:16175
if((esr(1)>0.0f)) {
// WRF:16176
if((ipconc>=3)) {
// WRF:16187
} else {
// WRF:16188
if((lwsm6&&(ipconc==0))) {
// WRF:16189
vt = vt2ave(1);
// WRF:16190
} else {
// WRF:16191
vt = vtxbar(1,ls,1);
// WRF:16192
}
// WRF:16194
qsacr(1) = lake_min(((((((lake_div(((0.25f)*pi),gf4))*esr(1))*qx(1,lr))*cx(1,ls))*lake_abs((vtxbar(1,lr,1)-vt)))*(((((gf6*gf1)*xdia(1,lr,2))+((((2.0f*gf5)*gf2)*xdia(1,lr,1))*xdia(1,ls,1)))+((gf4*gf3)*xdia(1,ls,2))))),qrmxd(1));
// WRF:16202
}
// WRF:16203
}
// WRF:16204
}
// WRF:16209
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16211
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16212
qhacw(1) = 0.0f;
// WRF:16213
qhacwmlr(1) = 0.0f;
// WRF:16214
rarx(1,lh) = 0.0f;
// WRF:16215
vhacw(1) = 0.0f;
// WRF:16216
vhsoak(1) = 0.0f;
// WRF:16217
zhacw(1) = 0.0f;
// WRF:16219
if(false) {
// WRF:16220
vtmax = ((gz(1,jgs,1)*dtpinv));
// WRF:16221
vtxbar(1,lh,1) = lake_min(vtmax,vtxbar(1,lh,1));
// WRF:16222
vtxbar(1,lh,2) = lake_min(vtmax,vtxbar(1,lh,2));
// WRF:16223
vtxbar(1,lh,3) = lake_min(vtmax,vtxbar(1,lh,3));
// WRF:16224
}
// WRF:16225
if((ehw(1)>0.0f)) {
// WRF:16227
if((ipconc>=2)) {
// WRF:16229
if(false) {
// WRF:16230
qhacw(1) = lake_div(((((((ehw(1)*((qx(1,lc)-qcwresv(1))))*cx(1,lh))*pi)*lake_abs((vtxbar(1,lh,1)-vtxbar(1,lc,1))))*((((2.0f*xdia(1,lh,1))*((xdia(1,lh,1)+(xdia(1,lc,1)*gf73rds))))+(xdia(1,lc,2)*gf83rds))))),4.f);
// WRF:16236
} else {
// WRF:16237
vt = lake_abs((vtxbar(1,lh,1)-vtxbar(1,lc,1)));
// WRF:16239
qhacw(1) = ((((((0.25f*pi)*ehw(1))*cx(1,lh))*((qx(1,lc)-qcwresv(1))))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab1lh(1,lh,lc)*xdia(1,lh,3))*xdia(1,lc,3)))+(da1lc(1)*lake_pow(xdia(1,lc,3),2)))));
// WRF:16244
}
// WRF:16245
qhacw(1) = lake_min(qhacw(1),((0.5f*qx(1,lc))*dtpinv));
// WRF:16247
if((lzh>1)) {
// WRF:16248
tmp = lake_div(qx(1,lh),cx(1,lh));
// WRF:16256
}
// WRF:16258
} else {
// WRF:16259
qhacw(1) = lake_min(((((((((0.25f)*pi))*ehw(1))*((qx(1,lc)-qcwresv(1))))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,lc,1))))*((((gf3*xdia(1,lh,2))+(((2.0f*gf2)*xdia(1,lh,1))*xdia(1,lc,1)))+(gf1*xdia(1,lc,2))))),((0.5f*((qx(1,lc)-qcwresv(1))))*dtpinv));
// WRF:16271
if(((lwsm6&&(qsacw(1)>0.0f))&&(qhacw(1)>0.0f))) {
// WRF:16272
qaacw = lake_div((((qx(1,ls)*qsacw(1))+(qx(1,lh)*qhacw(1)))),((qx(1,ls)+qx(1,lh))));
// WRF:16274
qsacw(1) = qaacw;
// WRF:16275
qhacw(1) = qaacw;
// WRF:16276
}
// WRF:16278
}
// WRF:16280
qhacwmlr(1) = qhacw(1);
// WRF:16281
if(((temg(1)>tfr)&&(iqhacwshr==0))) {
// WRF:16282
qhacw(1) = 0.0f;
// WRF:16283
}
// WRF:16285
if(((coeff_lvol{}(lh)>1)||(lhl>1))) {
// WRF:16287
if((temg(1)<273.15f)) {
// WRF:16288
if((irimdenopt==1)) {
// WRF:16289
vt = (((((1.0f-rimdenvwgt))*vtxbar(1,lh,1))+(rimdenvwgt*vtxbar(1,lh,2))));
// WRF:16291
rimdn(1,lh) = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*vt))),((temg(1)-273.15f)))),(rimc2)));
// WRF:16295
rimdn(1,lh) = lake_min(lake_max(rimc3,rimdn(1,lh)),rimc4);
// WRF:16305
} else if((irimdenopt==2)) {
// WRF:16307
tmp = (lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((((1.0f-rimdenvwgt))*vtxbar(1,lh,1))+(rimdenvwgt*vtxbar(1,lh,2))))),((temg(1)-273.15f))));
// WRF:16310
tmp = lake_min(lake_div(5.5f,0.6f),lake_max(lake_div(0.3f,0.6f),tmp));
// WRF:16312
rimdn(1,lh) = (1000.f*(((0.051f+(0.114f*tmp))-(0.0055f*lake_pow(tmp,2)))));
// WRF:16314
} else if(((irimdenopt==3)||(irimdenopt==4))) {
// WRF:16316
tmp = (lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((((1.0f-rimdenvwgt))*vtxbar(1,lh,1))+(rimdenvwgt*vtxbar(1,lh,2))))),((temg(1)-273.15f))));
// WRF:16321
if((irimdenopt==3)) {
// WRF:16322
rimdn(1,lh) = lake_min(900.f,lake_max(170.f,(110.f*lake_pow(tmp,0.76f))));
// WRF:16323
} else if((irimdenopt==4)) {
// WRF:16324
rimdn(1,lh) = lake_min(917.f,lake_max(10.f,(900.0f*((1.0f-lake_pow(0.905f,tmp))))));
// WRF:16325
}
// WRF:16327
}
// WRF:16328
} else {
// WRF:16329
rimdn(1,lh) = 1000.f;
// WRF:16330
}
// WRF:16332
if((coeff_lvol{}(lh)>1)) {vhacw(1) = lake_div((rho0(1)*qhacw(1)),rimdn(1,lh));}
// WRF:16334
}
// WRF:16336
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(ipelec>=1))) {
// WRF:16337
rarx(1,lh) = lake_div(((qhacw(1)*1.0e3f)*rho0(1)),((((lake_div(pi,2.0f))*xdia(1,lh,2))*cx(1,lh))));
// WRF:16339
}
// WRF:16341
}
// WRF:16342
}
// WRF:16345
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16346
qhaci(1) = 0.0f;
// WRF:16347
qhaci0(1) = 0.0f;
// WRF:16348
if((ehi(1)>0.0f)) {
// WRF:16349
if((ipconc>=5)) {
// WRF:16351
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,li,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,li,1))));
// WRF:16354
qhaci0(1) = ((((((0.25f*pi)*ehiclsn(1))*cx(1,lh))*qx(1,li))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab1lh(1,lh,li)*xdia(1,lh,3))*xdia(1,li,3)))+(coeff_da1{}(li)*lake_pow(xdia(1,li,3),2)))));
// WRF:16358
qhaci(1) = lake_min((ehi(1)*qhaci0(1)),qimxd(1));
// WRF:16359
} else {
// WRF:16360
qhaci(1) = lake_min((((((((((0.25f)*pi))*ehi(1))*ehiclsn(1))*qx(1,li))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,li,1))))*((((gf3*xdia(1,lh,2))+(((2.0f*gf2)*xdia(1,lh,1))*xdia(1,li,1)))+(gf1*xdia(1,li,2))))),qimxd(1));
// WRF:16368
}
// WRF:16369
}
// WRF:16370
}
// WRF:16373
if(((lis>1)&&(ipconc>=5))) {
// WRF:16374
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16375
qhacis(1) = 0.0f;
// WRF:16376
qhacis0(1) = 0.0f;
// WRF:16377
if((ehis(1)>0.0f)) {
// WRF:16379
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,lis,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,lis,1))));
// WRF:16382
qhacis0(1) = ((((((0.25f*pi)*ehisclsn(1))*cx(1,lh))*qx(1,lis))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab1lh(1,lh,lis)*xdia(1,lh,3))*xdia(1,lis,3)))+(coeff_da1{}(li)*lake_pow(xdia(1,lis,3),2)))));
// WRF:16386
qhacis(1) = lake_min((ehis(1)*qhacis0(1)),qxmxd(1,lis));
// WRF:16387
}
// WRF:16388
}
// WRF:16389
}
// WRF:16393
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16394
qhacs(1) = 0.0f;
// WRF:16395
qhacs0(1) = 0.0f;
// WRF:16396
if((ehs(1)>0.0f)) {
// WRF:16397
if((ipconc>=5)) {
// WRF:16399
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,ls,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,ls,1))));
// WRF:16402
qhacs0(1) = ((((((0.25f*pi)*ehsclsn(1))*cx(1,lh))*qx(1,ls))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab1lh(1,lh,ls)*xdia(1,lh,3))*xdia(1,ls,3)))+(coeff_da1{}(ls)*lake_pow(xdia(1,ls,3),2)))));
// WRF:16407
qhacs(1) = lake_min((ehs(1)*qhacs0(1)),qsmxd(1));
// WRF:16409
} else {
// WRF:16410
qhacs(1) = lake_min((((((((lake_div(((0.25f)*pi),gf4))*ehs(1))*ehsclsn(1))*qx(1,ls))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,ls,1))))*(((((gf6*gf1)*xdia(1,ls,2))+((((2.0f*gf5)*gf2)*xdia(1,ls,1))*xdia(1,lh,1)))+((gf4*gf3)*xdia(1,lh,2))))),qsmxd(1));
// WRF:16418
}
// WRF:16419
}
// WRF:16420
}
// WRF:16422
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16423
qhacr(1) = 0.0f;
// WRF:16424
qhacrmlr(1) = 0.0f;
// WRF:16425
vhacr(1) = 0.0f;
// WRF:16426
chacr(1) = 0.0f;
// WRF:16427
zhacr(1) = 0.0f;
// WRF:16428
if((temg(1)>tfr)) {raindn(1,lh) = 1000.0f;}
// WRF:16430
if((ehr(1)>0.0f)) {
// WRF:16431
if((ipconc>=3)) {
// WRF:16432
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,lr,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,lr,1))));
// WRF:16440
qhacr(1) = ((((((0.25f*pi)*ehr(1))*cx(1,lh))*qx(1,lr))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab1lh(1,lh,lr)*xdia(1,lh,3))*xdia(1,lr,3)))+(da1lr(1)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16450
qhacr(1) = lake_min(qhacr(1),qxmxd(1,lr));
// WRF:16452
qhacrmlr(1) = qhacr(1);
// WRF:16454
if(((temg(1)>tfr)&&(iehr0c==0))) {
// WRF:16455
qhacr(1) = 0.0f;
// WRF:16457
if((iqhacrmlr==0)) {
// WRF:16458
qhacrmlr(1) = (-qhacw(1));
// WRF:16459
}
// WRF:16461
} else {
// WRF:16470
chacr(1) = ((((((0.25f*pi)*ehr(1))*cx(1,lh))*cx(1,lr))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab0lh(1,lh,lr)*xdia(1,lh,3))*xdia(1,lr,3)))+(da0lr(1)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16478
chacr(1) = lake_min(chacr(1),crmxd(1));
// WRF:16480
if((lzh>1)) {
// WRF:16481
tmp = lake_div(qx(1,lh),cx(1,lh));
// WRF:16490
}
// WRF:16491
}
// WRF:16493
} else {
// WRF:16494
if((lwsm6&&(ipconc==0))) {
// WRF:16495
vt = vt2ave(1);
// WRF:16496
} else {
// WRF:16497
vt = vtxbar(1,lh,1);
// WRF:16498
}
// WRF:16500
qhacr(1) = lake_min(((((((lake_div(((0.25f)*pi),gf4))*ehr(1))*qx(1,lr))*cx(1,lh))*lake_abs((vt-vtxbar(1,lr,1))))*(((((gf6*gf1)*xdia(1,lr,2))+((((2.0f*gf5)*gf2)*xdia(1,lr,1))*xdia(1,lh,1)))+((gf4*gf3)*xdia(1,lh,2))))),qrmxd(1));
// WRF:16509
if((temg(1)>tfr)) {
// WRF:16510
if((iqhacrmlr>=1)) {qhacrmlr(1) = qhacr(1);}
// WRF:16511
qhacr(1) = 0.0f;
// WRF:16512
}
// WRF:16514
}
// WRF:16515
if(((coeff_lvol{}(lh)>1)||(lhl>1))) {
// WRF:16517
if((temg(1)<273.15f)) {
// WRF:16518
raindn(1,lh) = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lr,3))))*(((0.60f)*vt))),((temg(1)-273.15f)))),(rimc2)));
// WRF:16522
raindn(1,lh) = lake_min(lake_max(rimc3,raindn(1,lh)),rimc4);
// WRF:16523
} else {
// WRF:16524
raindn(1,lh) = 1000.f;
// WRF:16525
}
// WRF:16527
if((coeff_lvol{}(lh)>1)) {vhacr(1) = lake_div((rho0(1)*qhacr(1)),raindn(1,lh));}
// WRF:16528
}
// WRF:16529
}
// WRF:16530
}
// WRF:16534
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16537
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16538
qhlacw(1) = 0.0f;
// WRF:16539
qhlacwmlr(1) = 0.0f;
// WRF:16540
vhlacw(1) = 0.0f;
// WRF:16541
vhlsoak(1) = 0.0f;
// WRF:16542
if(((lhl>1)&&true)) {
// WRF:16543
vtmax = ((gz(1,jgs,1)*dtpinv));
// WRF:16544
vtxbar(1,lhl,1) = lake_min(vtmax,vtxbar(1,lhl,1));
// WRF:16545
vtxbar(1,lhl,2) = lake_min(vtmax,vtxbar(1,lhl,2));
// WRF:16546
vtxbar(1,lhl,3) = lake_min(vtmax,vtxbar(1,lhl,3));
// WRF:16547
}
// WRF:16549
if((lhl>0)) {
// WRF:16550
rarx(1,lhl) = 0.0f;
// WRF:16551
}
// WRF:16553
if(((lhl>1)&&(ehlw(1)>0.0f))) {
// WRF:16558
vt = lake_abs((vtxbar(1,lhl,1)-vtxbar(1,lc,1)));
// WRF:16560
qhlacw(1) = ((((((0.25f*pi)*ehlw(1))*cx(1,lhl))*((qx(1,lc)-qcwresv(1))))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((dab1lh(1,lhl,lc)*xdia(1,lhl,3))*xdia(1,lc,3)))+(da1lc(1)*lake_pow(xdia(1,lc,3),2)))));
// WRF:16566
qhlacw(1) = lake_min(qhlacw(1),((0.5f*qx(1,lc))*dtpinv));
// WRF:16568
qhlacwmlr(1) = qhlacw(1);
// WRF:16569
if(((temg(1)>tfr)&&(iqhlacwshr==0))) {
// WRF:16570
qhlacw(1) = 0.0f;
// WRF:16571
}
// WRF:16573
if((coeff_lvol{}(lhl)>1)) {
// WRF:16575
if((temg(1)<273.15f)) {
// WRF:16576
if((irimdenopt==1)) {
// WRF:16577
rimdn(1,lhl) = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*(((((1.0f-rimdenvwgt))*vtxbar(1,lhl,1))+(rimdenvwgt*vtxbar(1,lhl,2))))))),((temg(1)-273.15f)))),(rimc2)));
// WRF:16580
rimdn(1,lhl) = lake_min(lake_max(hldnmn,rimc3,rimdn(1,lhl)),rimc4);
// WRF:16582
} else if((irimdenopt==2)) {
// WRF:16583
tmp = lake_div(((((-0.5f)*(1.e+06f))*xdia(1,lc,1))*(((((1.0f-rimdenvwgt))*vtxbar(1,lhl,1))+(rimdenvwgt*vtxbar(1,lhl,2))))),((temg(1)-273.15f)));
// WRF:16586
tmp = lake_min(lake_div(5.5f,0.6f),lake_max(lake_div(0.3f,0.6f),tmp));
// WRF:16588
rimdn(1,lhl) = (1000.f*(((0.051f+(0.114f*tmp))-(0.005f*lake_pow(tmp,2)))));
// WRF:16590
} else if(((irimdenopt==3)||(irimdenopt==4))) {
// WRF:16591
tmp = lake_div(((((-0.5f)*(1.e+06f))*xdia(1,lc,1))*(((((1.0f-rimdenvwgt))*vtxbar(1,lhl,1))+(rimdenvwgt*vtxbar(1,lhl,2))))),((temg(1)-273.15f)));
// WRF:16596
if((irimdenopt==3)) {
// WRF:16597
rimdn(1,lhl) = lake_min(900.f,lake_max(170.f,(110.f*lake_pow(tmp,0.76f))));
// WRF:16598
} else if((irimdenopt==4)) {
// WRF:16599
rimdn(1,lhl) = lake_min(917.f,lake_max(10.f,(900.0f*((1.0f-lake_pow(0.905f,tmp))))));
// WRF:16600
}
// WRF:16602
}
// WRF:16603
} else {
// WRF:16604
rimdn(1,lhl) = 1000.f;
// WRF:16605
}
// WRF:16607
vhlacw(1) = lake_div((rho0(1)*qhlacw(1)),rimdn(1,lhl));
// WRF:16609
}
// WRF:16612
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(ipelec>=1))) {
// WRF:16613
rarx(1,lhl) = lake_div(((qhlacw(1)*1.0e3f)*rho0(1)),((((lake_div(pi,2.0f))*xdia(1,lhl,2))*cx(1,lhl))));
// WRF:16615
}
// WRF:16617
}
// WRF:16618
}
// WRF:16620
for(int _s0=1;_s0<=ngs;++_s0) {qhlaci(_s0) = 0.0f;}
// WRF:16621
for(int _s0=1;_s0<=ngs;++_s0) {qhlaci0(_s0) = 0.0f;}
// WRF:16622
if((lhl>1)) {
// WRF:16623
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16624
if((ehli(1)>0.0f)) {
// WRF:16625
if((ipconc>=5)) {
// WRF:16627
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,li,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,li,1))));
// WRF:16630
qhlaci0(1) = ((((((0.25f*pi)*ehliclsn(1))*cx(1,lhl))*qx(1,li))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((dab1lh(1,lhl,li)*xdia(1,lhl,3))*xdia(1,li,3)))+(coeff_da1{}(li)*lake_pow(xdia(1,li,3),2)))));
// WRF:16635
qhlaci(1) = lake_min((ehli(1)*qhlaci0(1)),qimxd(1));
// WRF:16636
}
// WRF:16637
}
// WRF:16638
}
// WRF:16639
}
// WRF:16641
for(int _s0=1;_s0<=ngs;++_s0) {qhlacs(_s0) = 0.0f;}
// WRF:16642
for(int _s0=1;_s0<=ngs;++_s0) {qhlacs0(_s0) = 0.0f;}
// WRF:16643
if((lhl>1)) {
// WRF:16644
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16645
if((ehls(1)>0.0f)) {
// WRF:16646
if((ipconc>=5)) {
// WRF:16648
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,ls,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,ls,1))));
// WRF:16651
qhlacs0(1) = ((((((0.25f*pi)*ehlsclsn(1))*cx(1,lhl))*qx(1,ls))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((dab1lh(1,lhl,ls)*xdia(1,lhl,3))*xdia(1,ls,3)))+(coeff_da1{}(ls)*lake_pow(xdia(1,ls,3),2)))));
// WRF:16656
qhlacs(1) = lake_min((ehls(1)*qhlacs0(1)),qsmxd(1));
// WRF:16657
}
// WRF:16658
}
// WRF:16659
}
// WRF:16660
}
// WRF:16663
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16664
qhlacr(1) = 0.0f;
// WRF:16665
qhlacrmlr(1) = 0.0f;
// WRF:16666
chlacr(1) = 0.0f;
// WRF:16667
vhlacr(1) = 0.0f;
// WRF:16668
if(((lhl>1)&&(temg(1)>tfr))) {raindn(1,lhl) = 1000.0f;}
// WRF:16670
if(((lhl>1)&&(ehlr(1)>0.0f))) {
// WRF:16671
if((ipconc>=3)) {
// WRF:16672
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,lr,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,lr,1))));
// WRF:16675
qhlacr(1) = ((((((0.25f*pi)*ehlr(1))*cx(1,lhl))*qx(1,lr))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((dab1lh(1,lhl,lr)*xdia(1,lhl,3))*xdia(1,lr,3)))+(da1lr(1)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16685
qhlacr(1) = lake_min(qhlacr(1),qxmxd(1,lr));
// WRF:16688
if((iqhlacrmlr>=1)) {qhlacrmlr(1) = qhlacr(1);}
// WRF:16690
if(((temg(1)>tfr)&&(iehlr0c==0))) {
// WRF:16691
qhlacr(1) = 0.0f;
// WRF:16692
if((iqhlacrmlr==0)) {
// WRF:16693
qhlacrmlr(1) = (-qhlacw(1));
// WRF:16694
}
// WRF:16695
} else {
// WRF:16696
chlacr(1) = ((((((0.25f*pi)*ehlr(1))*cx(1,lhl))*cx(1,lr))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((dab0lh(1,lhl,lr)*xdia(1,lhl,3))*xdia(1,lr,3)))+(da0lr(1)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16701
chlacr(1) = lake_min(chlacr(1),crmxd(1));
// WRF:16703
if((coeff_lvol{}(lhl)>1)) {
// WRF:16704
vhlacr(1) = lake_div((rho0(1)*qhlacr(1)),raindn(1,lhl));
// WRF:16705
}
// WRF:16706
}
// WRF:16707
}
// WRF:16708
}
// WRF:16709
}
// WRF:16719
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16721
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16722
qiacw(1) = 0.0f;
// WRF:16723
if((eiw(1)>0.0f)) {
// WRF:16725
vt = sqrtf((lake_pow(((vtxbar(1,li,1)-vtxbar(1,lc,1))),2)+((0.04f*vtxbar(1,li,1))*vtxbar(1,lc,1))));
// WRF:16728
qiacw(1) = ((((((0.25f*pi)*eiw(1))*cx(1,li))*qx(1,lc))*vt)*((((coeff_da0{}(li)*lake_pow(xdia(1,li,3),2))+((coeff_dab1{}(li,lc)*xdia(1,li,3))*xdia(1,lc,3)))+(da1lc(1)*lake_pow(xdia(1,lc,3),2)))));
// WRF:16733
qiacw(1) = lake_min(qiacw(1),qxmxd(1,lc));
// WRF:16734
}
// WRF:16735
}
// WRF:16740
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16742
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16743
qiacr(1) = 0.0f;
// WRF:16744
qiacrf(1) = 0.0f;
// WRF:16745
qiacrs(1) = 0.0f;
// WRF:16746
ciacrs(1) = 0.0f;
// WRF:16747
ciacr(1) = 0.0f;
// WRF:16748
ciacrf(1) = 0.0f;
// WRF:16749
viacrf(1) = 0.0f;
// WRF:16750
csplinter(1) = 0.0f;
// WRF:16751
qsplinter(1) = 0.0f;
// WRF:16752
csplinter2(1) = 0.0f;
// WRF:16753
qsplinter2(1) = 0.0f;
// WRF:16754
if((((iacr>=1)&&(eri(1)>0.0f))&&(temg(1)<=270.15f))) {
// WRF:16756
if((ipconc>=3)) {
// WRF:16757
ni = 0.0f;
// WRF:16758
if((xdia(1,li,1)>=10.e-6f)) {
// WRF:16759
ni = (ni+(cx(1,li)*gfk_exp((-lake_pow((lake_div(40.e-6f,xdia(1,li,1))),3)))));
// WRF:16760
}
// WRF:16761
if((imurain==1)) {
// WRF:16762
if((iacrsize!=4)) {
// WRF:16763
if((iacrsize==1)) {
// WRF:16764
ratio = lake_div(500.e-6f,xdia(1,lr,1));
// WRF:16765
} else if((iacrsize==2)) {
// WRF:16766
ratio = lake_div(300.e-6f,xdia(1,lr,1));
// WRF:16767
} else if((iacrsize==3)) {
// WRF:16768
ratio = lake_div(40.e-6f,xdia(1,lr,1));
// WRF:16769
} else if((iacrsize==5)) {
// WRF:16770
ratio = lake_div(150.e-6f,xdia(1,lr,1));
// WRF:16771
}
// WRF:16772
i = lake_min(nqiacrratio,((int)((ratio*dqiacrratioinv))));
// WRF:16773
j = ((int)((lake_max(0.0f,lake_min(15.f,alpha(1,lr)))*dqiacralphainv)));
// WRF:16775
delx = (ratio-(float(i)*dqiacrratio));
// WRF:16776
dely = (alpha(1,lr)-(float(j)*dqiacralpha));
// WRF:16777
ip1 = lake_min((i+1),nqiacrratio);
// WRF:16778
jp1 = lake_min((j+1),nqiacralpha);
// WRF:16781
tmp1 = (ciacrratio(i,j)+((delx*dqiacrratioinv)*((ciacrratio(ip1,j)-ciacrratio(i,j)))));
// WRF:16782
tmp2 = (ciacrratio(i,jp1)+((delx*dqiacrratioinv)*((ciacrratio(ip1,jp1)-ciacrratio(i,jp1)))));
// WRF:16786
nr = (((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*cx(1,lr));
// WRF:16789
tmp1 = (qiacrratio(i,j)+((delx*dqiacrratioinv)*((qiacrratio(ip1,j)-qiacrratio(i,j)))));
// WRF:16790
tmp2 = (qiacrratio(i,jp1)+((delx*dqiacrratioinv)*((qiacrratio(ip1,jp1)-qiacrratio(i,jp1)))));
// WRF:16794
qr = (((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*qx(1,lr));
// WRF:16796
} else {
// WRF:16797
nr = cx(1,lr);
// WRF:16798
qr = qx(1,lr);
// WRF:16799
}
// WRF:16801
vt = sqrtf((lake_pow(((vtxbar(1,lr,1)-vtxbar(1,li,1))),2)+((0.04f*vtxbar(1,lr,1))*vtxbar(1,li,1))));
// WRF:16804
qiacr(1) = ((((((0.25f*pi)*eri(1))*ni)*qr)*vt)*((((coeff_da0{}(li)*lake_pow(xdia(1,li,3),2))+((dab1lh(1,li,lr)*xdia(1,lr,3))*xdia(1,li,3)))+(coeff_da1{}(lr)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16809
qiacr(1) = lake_min(qrmxd(1),qiacr(1));
// WRF:16812
ciacr(1) = ((((((0.25f*pi)*eri(1))*ni)*nr)*vt)*((((coeff_da0{}(li)*lake_pow(xdia(1,li,3),2))+((dab0lh(1,li,lr)*xdia(1,lr,3))*xdia(1,li,3)))+(coeff_da0{}(lr)*lake_pow(xdia(1,lr,3),2)))));
// WRF:16817
ciacr(1) = lake_min(crmxd(1),ciacr(1));
// WRF:16824
} else if((imurain==3)) {
// WRF:16826
arg = (1000.f*xdia(1,lr,3));
// WRF:16829
if((ipconc>=3)) {
// WRF:16830
if((iacrsize==1)) {
// WRF:16831
nr = (cx(1,lr)*gaml02d500(arg));
// WRF:16832
} else if(((iacrsize==2)||(iacrsize==5))) {
// WRF:16833
nr = (cx(1,lr)*gaml02d300(arg));
// WRF:16834
} else if((iacrsize==3)) {
// WRF:16835
nr = (cx(1,lr)*gaml02(arg));
// WRF:16836
} else if((iacrsize==4)) {
// WRF:16837
nr = cx(1,lr);
// WRF:16838
}
// WRF:16839
} else {
// WRF:16840
nr = (cx(1,lr)*gaml02(arg));
// WRF:16841
}
// WRF:16845
if(((ni>0.0f)&&(nr>0.0f))) {
// WRF:16846
d0 = xdia(1,lr,3);
// WRF:16847
qiacr(1) = ((xdn(1,lr)*rhoinv(1))*((((0.217239f*((((((0.522295f*(lake_pow(d0,5)))+(49711.81f*(lake_pow(d0,6))))-(1.673016e7f*(lake_pow(d0,7))))+(2.404471e9f*(lake_pow(d0,8))))-(1.22872e11f*(lake_pow(d0,9))))))*ni)*nr)));
// WRF:16853
qiacr(1) = lake_min(qrmxd(1),qiacr(1));
// WRF:16854
ciacr(1) = ((((0.217239f*((((((0.2301947f*(lake_pow(d0,2)))+(15823.76f*(lake_pow(d0,3))))-(4.167685e6f*(lake_pow(d0,4))))+(4.920215e8f*(lake_pow(d0,5))))-(2.133344e10f*(lake_pow(d0,6))))))*ni)*nr));
// WRF:16860
ciacr(1) = lake_min(crmxd(1),ciacr(1));
// WRF:16862
}
// WRF:16863
}
// WRF:16864
if(((iacr==1)||(iacr==3))) {
// WRF:16865
ciacrf(1) = lake_min(ciacr(1),(lake_div(qiacr(1),(((1.0f*vr1mm)*1000.0f)))*rho0(1)));
// WRF:16866
} else if((iacr==2)) {
// WRF:16867
ciacrf(1) = ciacr(1);
// WRF:16868
} else if((iacr==4)) {
// WRF:16869
ciacrf(1) = lake_min(ciacr(1),(lake_div(qiacr(1),(((1.0f*vfrz)*1000.0f)))*rho0(1)));
// WRF:16870
} else if((iacr==5)) {
// WRF:16871
ciacrf(1) = (ciacr(1)*rzxh(1));
// WRF:16872
}
// WRF:16877
} else {
// WRF:16878
qiacr(1) = lake_min(((((((((lake_div(0.25f,gf4))*pi))*eri(1))*cx(1,li))*qx(1,lr))*lake_abs((vtxbar(1,lr,1)-vtxbar(1,li,1))))*(((((gf6*gf1)*xdia(1,lr,2))+((((2.0f*gf5)*gf2)*xdia(1,lr,1))*xdia(1,li,1)))+((gf4*gf3)*xdia(1,li,2))))),qrmxd(1));
// WRF:16886
}
// WRF:16887
}
// WRF:16892
if((ipconc>=1)) {
// WRF:16893
if((nsplinter>=1000)) {
// WRF:16896
if((((qiacr(1)*dtp)>coeff_qxmin{}(lh))&&(ciacr(1)>1.e-3f))) {
// WRF:16897
tmpdiam = (1.e6f*lake_pow((lake_div((6.f*qiacr(1)),(((1000.f*pi)*ciacr(1))))),(lake_div(1.f,3.f))));
// WRF:16898
csplinter(1) = ((lawson_splinter_fac*lake_pow(tmpdiam,4))*ciacr(1));
// WRF:16899
}
// WRF:16900
} else if((nsplinter>=0)) {
// WRF:16901
csplinter(1) = (nsplinter*ciacr(1));
// WRF:16902
} else {
// WRF:16903
csplinter(1) = ((-nsplinter)*ciacrf(1));
// WRF:16904
}
// WRF:16905
qsplinter(1) = lake_min((0.1f*qiacr(1)),lake_div((csplinter(1)*splintermass),rho0(1)));
// WRF:16906
}
// WRF:16908
frach = 1.0f;
// WRF:16909
if(((ibiggsnow==2)||(ibiggsnow==3))) {
// WRF:16910
if((ciacr(1)>coeff_qxmin{}(lh))) {
// WRF:16911
xvfrz = lake_div((rho0(1)*qiacr(1)),((ciacr(1)*900.f)));
// WRF:16912
frach = (0.5f*((1.f+tanhf((0.2e12f*((xvfrz-(1.15f*xvbiggsnow))))))));
// WRF:16914
qiacrs(1) = (((1.f-frach))*qiacr(1));
// WRF:16915
ciacrs(1) = (((1.f-frach))*ciacrf(1));
// WRF:16917
}
// WRF:16918
}
// WRF:16920
qiacrf(1) = (frach*qiacr(1));
// WRF:16921
ciacrf(1) = (frach*ciacrf(1));
// WRF:16923
if((coeff_lvol{}(lh)>1)) {
// WRF:16924
viacrf(1) = lake_div((rho0(1)*qiacrf(1)),rhofrz);
// WRF:16925
}
// WRF:16927
}
// WRF:16934
if((ipconc>=4)) {
// WRF:16935
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16936
csacs(1) = 0.0f;
// WRF:16937
if(((qx(1,ls)>coeff_qxmin{}(ls))&&(ess(1)>0.0f))) {
// WRF:16939
if((iessec0flag==0)) {
// WRF:16940
ec0(1) = 1.0f;
// WRF:16941
} else {
// WRF:16942
tmp = lake_div(xv(1,ls),((coeff_xvmx{}(ls)*lake_max(1.f,lake_div(100.f,lake_min(100.f,xdn(1,ls)))))));
// WRF:16943
if((tmp<essfrac1)) {
// WRF:16944
ec0(1) = 1.0f;
// WRF:16945
} else if((tmp>=essfrac2)) {
// WRF:16946
ec0(1) = 0.0f;
// WRF:16947
} else {
// WRF:16948
ec0(1) = lake_div(((essfrac2-tmp)),((essfrac2-essfrac1)));
// WRF:16949
}
// WRF:16950
}
// WRF:16952
csacs(1) = (((((ec0(1)*rvt)*aa2)*ess(1))*lake_pow(cx(1,ls),2))*lake_min(xv(1,ls),(lake_div((4.f*pii),3.f)*lake_pow(essrmax,3))));
// WRF:16954
csacs(1) = lake_min(csacs(1),csmxd(1));
// WRF:16955
}
// WRF:16956
}
// WRF:16957
}
// WRF:16960
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16961
if(((ipconc>=2)||(ipelec>=9))) {
// WRF:16962
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16963
ciacw(1) = 0.0f;
// WRF:16964
if((eiw(1)>0.0f)) {
// WRF:16965
ciacw(1) = lake_div((qiacw(1)*rho0(1)),xmas(1,lc));
// WRF:16966
ciacw(1) = lake_min(ciacw(1),ccmxd(1));
// WRF:16967
}
// WRF:16968
}
// WRF:16970
}
// WRF:16972
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:16973
if(((ipconc>=2)||(ipelec>=1))) {
// WRF:16974
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:16975
cracw(1) = 0.0f;
// WRF:16976
cracr(1) = 0.0f;
// WRF:16977
ec0(1) = 1.e9f;
// WRF:16978
if((((qx(1,lc)>coeff_qxmin{}(lc))&&(qx(1,lr)>coeff_qxmin{}(lr)))&&(qracw(1)>0.0f))) {
// WRF:16981
if((ipconc<3)) {
// WRF:16982
if((erw(1)>0.0f)) {
// WRF:16983
cracw(1) = ((((((((0.25f)*pi))*erw(1))*((cx(1,lc)-ccwresv(1))))*cx(1,lr))*lake_abs((vtxbar(1,lr,1)-vtxbar(1,lc,1))))*((((gf1*xdia(1,lc,2))+(((2.0f*gf2)*xdia(1,lc,1))*xdia(1,lr,1)))+(gf3*xdia(1,lr,2)))));
// WRF:16989
}
// WRF:16990
} else {
// WRF:16991
if(((dmrauto<=0)||((rho0(1)*qx(1,lr))>(1.2f*xl2p(1))))) {
// WRF:16992
if(((0.5f*xdia(1,lr,3))>rh(1))) {
// WRF:16994
if(((0.5f*xdia(1,lr,3))>rwradmn)) {
// WRF:16997
cracw(1) = (((aa2*cx(1,lr))*((cx(1,lc)-ccwresv(1))))*((xv(1,lc)+xv(1,lr))));
// WRF:16998
} else {
// WRF:16999
if((imurain==3)) {
// WRF:17001
cracw(1) = (((aa1*cx(1,lr))*((cx(1,lc)-ccwresv(1))))*((lake_div((((alpha(1,lc)+2.f))*lake_pow(xv(1,lc),2)),((alpha(1,lc)+1.f)))+lake_div((((alpha(1,lr)+2.f))*lake_pow(xv(1,lr),2)),((alpha(1,lr)+1.f))))));
// WRF:17004
} else {
// WRF:17005
cracw(1) = (((aa1*cx(1,lr))*((cx(1,lc)-ccwresv(1))))*((lake_div((((alpha(1,lc)+2.f))*lake_pow(xv(1,lc),2)),((alpha(1,lc)+1.f)))+lake_div((((((alpha(1,lr)+6.f))*((alpha(1,lr)+5.f)))*((alpha(1,lr)+4.f)))*lake_pow(xv(1,lr),2)),(((((alpha(1,lr)+3.f))*((alpha(1,lr)+2.f)))*((alpha(1,lr)+1.f))))))));
// WRF:17009
}
// WRF:17010
}
// WRF:17011
}
// WRF:17012
}
// WRF:17013
}
// WRF:17014
}
// WRF:17019
ec0(1) = 2.e9f;
// WRF:17020
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:17021
rwrad = (0.5f*xdia(1,lr,3));
// WRF:17025
if((icracrthresh>1)) {
// WRF:17026
if((imurain==1)) {
// WRF:17027
tmp = (((3.67f+alpha(1,lr)))*xdia(1,lr,1));
// WRF:17028
} else {
// WRF:17029
tmp = (lake_pow(((1.678f+alpha(1,lr))),(lake_div(1.f,3.f)))*xdia(1,lr,1));
// WRF:17030
}
// WRF:17031
} else {
// WRF:17032
tmp = (xdia(1,lr,3)-0.1e-3f);
// WRF:17033
}
// WRF:17036
if(((tmp>1.9e-3f)||(icracr<=0))) {
// WRF:17037
ec0(1) = 0.0f;
// WRF:17038
cracr(1) = 0.0f;
// WRF:17039
} else {
// WRF:17040
if(((dmrauto<=0)||((rho0(1)*qx(1,lr))>(1.2f*xl2p(1))))) {
// WRF:17041
if((xdia(1,lr,3)<6.1e-4f)) {
// WRF:17042
ec0(1) = 1.0f;
// WRF:17043
} else {
// WRF:17044
ec0(1) = gfk_exp(((-50.0f)*((50.0f*((xdia(1,lr,3)-6.0e-4f))))));
// WRF:17045
}
// WRF:17048
if((rwrad>=50.e-6f)) {
// WRF:17049
cracr(1) = (((ec0(1)*aa2)*lake_pow(cx(1,lr),2))*xv(1,lr));
// WRF:17050
} else {
// WRF:17051
if((imurain==3)) {
// WRF:17052
cracr(1) = lake_div((((ec0(1)*aa1)*lake_pow(((cx(1,lr)*xv(1,lr))),2))*((alpha(1,lr)+2.f))),((alpha(1,lr)+1.f)));
// WRF:17054
} else {
// WRF:17055
cracr(1) = lake_div((((((ec0(1)*aa1)*lake_pow(((cx(1,lr)*xv(1,lr))),2))*((alpha(1,lr)+6.f)))*((alpha(1,lr)+5.f)))*((alpha(1,lr)+4.f))),(((((alpha(1,lr)+3.f))*((alpha(1,lr)+2.f)))*((alpha(1,lr)+1.f)))));
// WRF:17059
}
// WRF:17060
}
// WRF:17062
}
// WRF:17063
}
// WRF:17064
}
// WRF:17067
}
// WRF:17068
}
// WRF:17075
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17076
for(int _s0=1;_s0<=ngs;++_s0) {chacw(_s0) = 0.0f;}
// WRF:17077
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17078
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17080
if((ipconc>=5)) {
// WRF:17081
if(((qhacw(1)>0.0f)&&(xmas(1,lc)>0.0f))) {
// WRF:17095
chacw(1) = lake_div((qhacw(1)*rho0(1)),xmascw(1));
// WRF:17097
chacw(1) = lake_min(chacw(1),((0.5f*((cx(1,lc)-ccwresv(1))))*dtpinv));
// WRF:17098
} else {
// WRF:17099
qhacw(1) = 0.0f;
// WRF:17100
}
// WRF:17101
} else {
// WRF:17103
chacw(1) = ((((((((0.25f)*pi))*ehw(1))*cx(1,lc))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,lc,1))))*((((gf1*xdia(1,lc,2))+(((2.0f*gf2)*xdia(1,lc,1))*xdia(1,lh,1)))+(gf3*xdia(1,lh,2)))));
// WRF:17109
chacw(1) = lake_min(chacw(1),((0.5f*cx(1,lc))*dtpinv));
// WRF:17112
}
// WRF:17113
}
// WRF:17114
}
// WRF:17116
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17117
for(int _s0=1;_s0<=ngs;++_s0) {chaci(_s0) = 0.0f;}
// WRF:17118
for(int _s0=1;_s0<=ngs;++_s0) {chaci0(_s0) = 0.0f;}
// WRF:17119
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17120
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17121
if(((ehi(1)>0.0f)||(((ehiclsn(1)>0.0f)&&(ipelec>0))))) {
// WRF:17122
if((ipconc>=5)) {
// WRF:17124
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,li,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,li,1))));
// WRF:17127
chaci0(1) = ((((((0.25f*pi)*ehiclsn(1))*cx(1,lh))*cx(1,li))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab0lh(1,lh,li)*xdia(1,lh,3))*xdia(1,li,3)))+(coeff_da0{}(li)*lake_pow(xdia(1,li,3),2)))));
// WRF:17132
} else {
// WRF:17133
chaci0(1) = ((((((((0.25f)*pi))*ehiclsn(1))*cx(1,li))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,li,1))))*((((gf1*xdia(1,li,2))+(((2.0f*gf2)*xdia(1,li,1))*xdia(1,lh,1)))+(gf3*xdia(1,lh,2)))));
// WRF:17139
}
// WRF:17141
chaci(1) = lake_min((ehi(1)*chaci0(1)),cimxd(1));
// WRF:17142
}
// WRF:17143
}
// WRF:17144
}
// WRF:17147
for(int _s0=1;_s0<=ngs;++_s0) {chacis(_s0) = 0.0f;}
// WRF:17148
if((((lis>1)&&(ipconc>=5))||(ipelec>=1))) {
// WRF:17149
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17150
if(((ehis(1)>0.0f)||(((ehisclsn(1)>0.0f)&&(ipelec>0))))) {
// WRF:17152
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,lis,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,lis,1))));
// WRF:17155
chacis0(1) = ((((((0.25f*pi)*ehisclsn(1))*cx(1,lh))*cx(1,lis))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab0lh(1,lh,lis)*xdia(1,lh,3))*xdia(1,lis,3)))+(coeff_da0{}(lis)*lake_pow(xdia(1,lis,3),2)))));
// WRF:17161
chacis(1) = lake_min((ehis(1)*chacis0(1)),cxmxd(1,lis));
// WRF:17162
}
// WRF:17163
}
// WRF:17164
}
// WRF:17167
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17168
for(int _s0=1;_s0<=ngs;++_s0) {chacs(_s0) = 0.0f;}
// WRF:17169
for(int _s0=1;_s0<=ngs;++_s0) {chacs0(_s0) = 0.0f;}
// WRF:17170
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17171
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17172
if((ehs(1)>0)) {
// WRF:17173
if(((ipconc>=5)||(((ehsclsn(1)>0.0f)&&(ipelec>0))))) {
// WRF:17175
vt = sqrtf((lake_pow(((vtxbar(1,lh,1)-vtxbar(1,ls,1))),2)+((0.04f*vtxbar(1,lh,1))*vtxbar(1,ls,1))));
// WRF:17178
chacs0(1) = ((((((0.25f*pi)*ehsclsn(1))*cx(1,lh))*cx(1,ls))*vt)*((((da0lh(1)*lake_pow(xdia(1,lh,3),2))+((dab0lh(1,lh,ls)*xdia(1,lh,3))*xdia(1,ls,3)))+(coeff_da0{}(ls)*lake_pow(xdia(1,ls,3),2)))));
// WRF:17183
} else {
// WRF:17184
chacs0(1) = ((((((((0.25f)*pi))*ehsclsn(1))*cx(1,ls))*cx(1,lh))*lake_abs((vtxbar(1,lh,1)-vtxbar(1,ls,1))))*(((((gf3*gf1)*xdia(1,ls,2))+((((2.0f*gf2)*gf2)*xdia(1,ls,1))*xdia(1,lh,1)))+((gf1*gf3)*xdia(1,lh,2)))));
// WRF:17190
}
// WRF:17191
chacs(1) = lake_min((ehs(1)*chacs0(1)),csmxd(1));
// WRF:17192
}
// WRF:17193
}
// WRF:17194
}
// WRF:17201
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17202
for(int _s0=1;_s0<=ngs;++_s0) {chlacw(_s0) = 0.0f;}
// WRF:17203
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17204
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17206
if(((lhl>1)&&(ipconc>=5))) {
// WRF:17207
if(((qhlacw(1)>0.0f)&&(xmas(1,lc)>0.0f))) {
// WRF:17221
chlacw(1) = lake_div((qhlacw(1)*rho0(1)),xmascw(1));
// WRF:17223
chlacw(1) = lake_min(chlacw(1),((0.5f*cx(1,lc))*dtpinv));
// WRF:17224
} else {
// WRF:17225
qhlacw(1) = 0.0f;
// WRF:17226
}
// WRF:17237
}
// WRF:17238
}
// WRF:17239
}
// WRF:17241
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17242
for(int _s0=1;_s0<=ngs;++_s0) {chlaci(_s0) = 0.0f;}
// WRF:17243
for(int _s0=1;_s0<=ngs;++_s0) {chlaci0(_s0) = 0.0f;}
// WRF:17244
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17245
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17246
if(((lhl>1)&&(((ehli(1)>0.0f)||(((ipelec>0)&&(ehliclsn(1)>0.0f))))))) {
// WRF:17247
if((ipconc>=5)) {
// WRF:17249
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,li,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,li,1))));
// WRF:17252
chlaci0(1) = ((((((0.25f*pi)*ehliclsn(1))*cx(1,lhl))*cx(1,li))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((coeff_dab0{}(lhl,li)*xdia(1,lhl,3))*xdia(1,li,3)))+(coeff_da0{}(li)*lake_pow(xdia(1,li,3),2)))));
// WRF:17264
}
// WRF:17266
chlaci(1) = lake_min((ehli(1)*chlaci0(1)),cimxd(1));
// WRF:17267
}
// WRF:17268
}
// WRF:17269
}
// WRF:17272
if(((lis>1)&&(ipconc>=5))) {
// WRF:17274
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17275
for(int _s0=1;_s0<=ngs;++_s0) {chlacis(_s0) = 0.0f;}
// WRF:17276
for(int _s0=1;_s0<=ngs;++_s0) {chlacis0(_s0) = 0.0f;}
// WRF:17277
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17278
if(((lhl>1)&&(((ehlis(1)>0.0f)||(((ipelec>0)&&(ehlisclsn(1)>0.0f))))))) {
// WRF:17280
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,lis,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,lis,1))));
// WRF:17283
chlacis0(1) = ((((((0.25f*pi)*ehlisclsn(1))*cx(1,lhl))*cx(1,lis))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((coeff_dab0{}(lhl,lis)*xdia(1,lhl,3))*xdia(1,lis,3)))+(coeff_da0{}(lis)*lake_pow(xdia(1,lis,3),2)))));
// WRF:17289
chlacis(1) = lake_min((ehlis(1)*chlacis0(1)),cxmxd(1,lis));
// WRF:17290
}
// WRF:17291
}
// WRF:17292
}
// WRF:17296
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17297
for(int _s0=1;_s0<=ngs;++_s0) {chlacs(_s0) = 0.0f;}
// WRF:17298
for(int _s0=1;_s0<=ngs;++_s0) {chlacs0(_s0) = 0.0f;}
// WRF:17299
if(((ipconc>=1)||(ipelec>=1))) {
// WRF:17300
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17301
if(((lhl>1)&&(((ehls(1)>0.0f)||(((ipelec>0)&&(ehlsclsn(1)>0.0f))))))) {
// WRF:17302
if((ipconc>=5)) {
// WRF:17304
vt = sqrtf((lake_pow(((vtxbar(1,lhl,1)-vtxbar(1,ls,1))),2)+((0.04f*vtxbar(1,lhl,1))*vtxbar(1,ls,1))));
// WRF:17307
chlacs0(1) = ((((((0.25f*pi)*ehlsclsn(1))*cx(1,lhl))*cx(1,ls))*vt)*((((da0lhl(1)*lake_pow(xdia(1,lhl,3),2))+((coeff_dab0{}(lhl,ls)*xdia(1,lhl,3))*xdia(1,ls,3)))+(coeff_da0{}(ls)*lake_pow(xdia(1,ls,3),2)))));
// WRF:17319
}
// WRF:17320
chlacs(1) = lake_min((ehls(1)*chlacs0(1)),csmxd(1));
// WRF:17321
}
// WRF:17322
}
// WRF:17323
}
// WRF:17329
if((ipconc>=2)) {
// WRF:17330
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17332
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17333
zrcnw(1) = 0.0f;
// WRF:17334
qrcnw(1) = 0.0f;
// WRF:17335
crcnw(1) = 0.0f;
// WRF:17336
cautn(1) = 0.0f;
// WRF:17337
}
// WRF:17339
if((dmrauto>=(-1))) {
// WRF:17340
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17343
if((((qx(1,lc)>coeff_qxmin{}(lc))&&(cx(1,lc)>1000.f))&&(temg(1)>(tfrh+4.f)))) {
// WRF:17345
volb = (xv(1,lc)*lake_pow((lake_div(1.f,((1.f+alpha(1,lc))))),(lake_div(1.f,2.f))));
// WRF:17346
cautn(1) = lake_min(ccmxd(1),((((lake_div(((alpha(1,lc)+2.f)),((alpha(1,lc)+1.f))))*aa1)*lake_pow(cx(1,lc),2))*lake_pow(xv(1,lc),2)));
// WRF:17348
cautn(1) = lake_max(0.0e0,cautn(1));
// WRF:17349
if(((rb(1)<=7.51e-6)||(dmrauto==(-1)))) {
// WRF:17350
t2s = 1.e30;
// WRF:17352
} else {
// WRF:17358
t2s = lake_div(3.72f,((((1.e6f*((rb(1)-7.500e-6)))*rho0(1))*qx(1,lc))));
// WRF:17360
qrcnw(1) = lake_max(0.0e0,lake_div(xl2p(1),((t2s*rho0(1)))));
// WRF:17361
crcnw(1) = lake_max(0.0e0,lake_min(lake_div((3.5e9f*xl2p(1)),t2s),(0.5f*cautn(1))));
// WRF:17363
if((dmrauto==0)) {
// WRF:17364
if((((qx(1,lr)*rho0(1))>(1.2f*xl2p(1)))&&(cx(1,lr)>cxmin))) {
// WRF:17365
crcnw(1) = (lake_div(cx(1,lr),qx(1,lr))*qrcnw(1));
// WRF:17366
} else if(((((dmropt==1)||(dmropt==3)))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17367
tmp = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17368
crcnw(1) = lake_min(tmp,crcnw(1));
// WRF:17369
} else if((((dmropt==4))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17370
tmp = crcnw(1);
// WRF:17371
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17373
crcnw(1) = lake_div((((tmp*qrcnw(1))+(tmp2*qx(1,lr)))),((qrcnw(1)+qx(1,lr))));
// WRF:17374
} else if((((dmropt==5))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17375
tmp = crcnw(1);
// WRF:17376
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17378
crcnw(1) = lake_div((((tmp*qx(1,lc))+(tmp2*qx(1,lr)))),((qx(1,lc)+qx(1,lr))));
// WRF:17379
} else if((((dmropt==6))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17380
tmp = crcnw(1);
// WRF:17381
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17383
crcnw(1) = lake_div(((((tmp*xdia(1,lc,3))*qx(1,lc))+((tmp2*xdia(1,lr,3))*qx(1,lr)))),(((xdia(1,lc,3)*qx(1,lc))+(xdia(1,lr,3)*qx(1,lr)))));
// WRF:17384
} else if((((dmropt==7))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17385
tmp = crcnw(1);
// WRF:17386
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17388
crcnw(1) = lake_div((((tmp*xdia(1,lc,3))+(tmp2*xdia(1,lr,3)))),((xdia(1,lc,3)+xdia(1,lr,3))));
// WRF:17389
} else if((((dmropt==8))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:17390
tmp = crcnw(1);
// WRF:17391
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17393
crcnw(1) = lake_div((((tmp*sqrtf(xdia(1,lc,3)))+(tmp2*sqrtf(xdia(1,lr,3))))),((sqrtf(xdia(1,lc,3))+sqrtf(xdia(1,lr,3)))));
// WRF:17394
}
// WRF:17395
} else if(((dmrauto==1)&&(cx(1,lr)>cxmin))) {
// WRF:17396
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:17397
tmp = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17398
crcnw(1) = lake_min(tmp,crcnw(1));
// WRF:17399
}
// WRF:17400
} else if(((dmrauto==2)&&(cx(1,lr)>cxmin))) {
// WRF:17401
tmp = crcnw(1);
// WRF:17402
tmp2 = lake_div((qrcnw(1)*cx(1,lr)),qx(1,lr));
// WRF:17404
crcnw(1) = lake_div((((tmp*qrcnw(1))+(tmp2*qx(1,lr)))),((qrcnw(1)+qx(1,lr))));
// WRF:17405
} else if(((dmrauto==3)&&(cx(1,lr)>cxmin))) {
// WRF:17406
tmp = lake_max((2.e0*rh(1)),((double)(xdia(1,lr,3))));
// WRF:17407
crcnw(1) = lake_div((rho0(1)*qrcnw(1)),(((lake_div(pi,6.f)*1000.f)*lake_pow(tmp,3))));
// WRF:17408
}
// WRF:17410
if((crcnw(1)<1.e-30f)) {qrcnw(1) = 0.0f;}
// WRF:17412
if((ipconc>=6)) {
// WRF:17413
if(((lzr>1)&&(qrcnw(1)>0.0f))) {
// WRF:17419
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(((dmrauto==1)||(dmrauto==2))))) {
// WRF:17420
tmp3 = lake_div(qx(1,lr),cx(1,lr));
// WRF:17421
tmp4 = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*((((2.f*tmp3)*qrcnw(1))-(lake_pow(tmp3,2)*crcnw(1)))));
// WRF:17423
if((imurain==3)) {
// WRF:17424
vr = lake_div((rho0(1)*qrcnw(1)),(1000.f));
// WRF:17425
tmp3 = lake_div(((36.f*((coeff_xnu{}(lc)+2.0f)))*lake_pow(vr,2)),(((crcnw(1)*((coeff_xnu{}(lc)+1.0f)))*lake_pow(pi,2))));
// WRF:17426
} else {
// WRF:17427
tmp3 = lake_div((galpharaut*lake_pow((lake_div(((6.f*rho0(1))*qrcnw(1)),((pi*coeff_xdn0{}(lr))))),2)),crcnw(1));
// WRF:17428
}
// WRF:17429
if((dmrauto==1)) {
// WRF:17430
zrcnw(1) = tmp4;
// WRF:17431
} else if((dmrauto==2)) {
// WRF:17432
zrcnw(1) = lake_div((((tmp3*qrcnw(1))+(tmp4*qx(1,lr)))),((qrcnw(1)+qx(1,lr))));
// WRF:17433
}
// WRF:17434
} else {
// WRF:17435
if((imurain==3)) {
// WRF:17436
vr = lake_div((rho0(1)*qrcnw(1)),(1000.f));
// WRF:17437
zrcnw(1) = lake_div(((36.f*((coeff_xnu{}(lc)+2.0f)))*lake_pow(vr,2)),(((crcnw(1)*((coeff_xnu{}(lc)+1.0f)))*lake_pow(pi,2))));
// WRF:17438
} else {
// WRF:17439
if((((dmropt<=1)||(dmropt>=4))||(((qx(1,lr)<coeff_qxmin{}(lr))&&(cx(1,lr)<cxmin))))) {
// WRF:17440
zrcnw(1) = lake_div((galpharaut*lake_pow((lake_div(((6.f*rho0(1))*qrcnw(1)),((pi*coeff_xdn0{}(lr))))),2)),crcnw(1));
// WRF:17441
} else {
// WRF:17442
tmp3 = lake_div(qx(1,lr),cx(1,lr));
// WRF:17443
zrcnw(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*((((2.f*tmp3)*qrcnw(1))-(lake_pow(tmp3,2)*crcnw(1)))));
// WRF:17445
}
// WRF:17448
}
// WRF:17449
}
// WRF:17451
}
// WRF:17452
}
// WRF:17477
}
// WRF:17480
}
// WRF:17481
}
// WRF:17483
}
// WRF:17487
} else {
// WRF:17493
if((ircnw==4)) {
// WRF:17494
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17496
qrcnw(1) = 0.0f;
// WRF:17497
qdiff = lake_max(((qx(1,lc)-qminrncw)),0.0f);
// WRF:17498
if(((qdiff>0.0f)&&(xdia(1,lc,1)>20.0e-6f))) {
// WRF:17499
argrcnw = (((1.2e-4f)+lake_div(((1.596e-12f)*((cx(1,lc)*1.0e-6f))),((((cwdisp*qdiff)*1.0e-3f)*rho0(1))))));
// WRF:17502
qrcnw(1) = lake_div((((rho0(1)*1e-3f))*(lake_pow(qdiff,2))),argrcnw);
// WRF:17504
qrcnw(1) = (lake_max(qrcnw(1),0.0f));
// WRF:17505
}
// WRF:17506
}
// WRF:17508
}
// WRF:17515
if((ircnw==5)) {
// WRF:17516
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17517
qrcnw(1) = 0.0f;
// WRF:17518
qrcnw(1) = 0.0f;
// WRF:17519
qccrit = lake_div((((lake_div(pi,6.f))*((cx(1,lc)*lake_pow(cwdiap,3))))*xdn(1,lc)),rho0(1));
// WRF:17520
qdiff = lake_max(((qx(1,lc)-qccrit)),0.f);
// WRF:17521
if(((qdiff>0.0f)&&(cx(1,lc)>1.0f))) {
// WRF:17522
argrcnw = (((1.2e-4f)+lake_div((((1.596e-12f)*cx(1,lc))*1.0e-3f),(((cwdisp*rho0(1))*qdiff)))));
// WRF:17525
qrcnw(1) = lake_div(((1.0e-3f*rho0(1))*(lake_pow(qdiff,2))),argrcnw);
// WRF:17528
qrcnw(1) = lake_min(qxmxd(1,lc),(lake_max(qrcnw(1),0.0f)));
// WRF:17531
}
// WRF:17532
}
// WRF:17533
}
// WRF:17539
if((ircnw==2)) {
// WRF:17540
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17541
qrcnw(1) = 0.0f;
// WRF:17542
qrcnw(1) = ((0.001f)*lake_max(((qx(1,lc)-qminrncw)),0.0f));
// WRF:17543
}
// WRF:17544
}
// WRF:17550
if((ircnw==1)) {
// WRF:17551
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17552
qrcnw(1) = 0.0f;
// WRF:17553
c1 = 0.2f;
// WRF:17554
c4 = lake_div(pi,(6.0f));
// WRF:17555
bradp = (((1.e+06f)*(lake_pow((lake_div(c1,(0.38f))),(lake_div(1.f,3.f)))))*((xdia(1,lc,1)*(0.5f))));
// WRF:17557
bl2 = ((0.027f)*(((((100.0f)*(lake_pow(bradp,3)))*((xdia(1,lc,1)*(0.5f))))-(0.4f))));
// WRF:17559
bt2 = lake_div(((bradp-7.5f)),(3.72f));
// WRF:17560
qrcnw(1) = 0.0f;
// WRF:17561
if(((bl2>0.0f)&&(bt2>0.0f))) {
// WRF:17562
qrcnw(1) = ((((bl2*bt2)*rho0(1))*qx(1,lc))*qx(1,lc));
// WRF:17564
}
// WRF:17565
}
// WRF:17566
}
// WRF:17570
}
// WRF:17577
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17578
for(int _s0=1;_s0<=ngs;++_s0) {qrfrz(_s0) = 0.0f;}
// WRF:17579
for(int _s0=1;_s0<=ngs;++_s0) {qrfrzs(_s0) = 0.0f;}
// WRF:17580
for(int _s0=1;_s0<=ngs;++_s0) {qrfrzf(_s0) = 0.0f;}
// WRF:17581
for(int _s0=1;_s0<=ngs;++_s0) {vrfrzf(_s0) = 0.0f;}
// WRF:17582
for(int _s0=1;_s0<=ngs;++_s0) {crfrz(_s0) = 0.0f;}
// WRF:17583
for(int _s0=1;_s0<=ngs;++_s0) {crfrzs(_s0) = 0.0f;}
// WRF:17584
for(int _s0=1;_s0<=ngs;++_s0) {crfrzf(_s0) = 0.0f;}
// WRF:17585
for(int _s0=1;_s0<=ngs;++_s0) {zrfrz(_s0) = 0.0f;}
// WRF:17586
for(int _s0=1;_s0<=ngs;++_s0) {zrfrzs(_s0) = 0.0f;}
// WRF:17587
for(int _s0=1;_s0<=ngs;++_s0) {zrfrzf(_s0) = 0.0f;}
// WRF:17588
for(int _s0=1;_s0<=ngs;++_s0) {qwcnr(_s0) = 0.0f;}
// WRF:17590
if((!(((ipconc==0)&&lwsm6)))) {
// WRF:17592
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17593
if((((qx(1,lr)>coeff_qxmin{}(lr))&&(temcg(1)<(-5.f)))&&(ibiggopt>0))) {
// WRF:17596
if((ipconc<3)) {
// WRF:17597
qrfrz(1) = lake_min((((((((20.0f)*(lake_pow(pi,2)))*brz)*(lake_div(xdn(1,lr),rho0(1))))*cx(1,lr))*(lake_pow(xdia(1,lr,1),6)))*((gfk_exp(lake_max(((-arz)*temcg(1)),0.0f))-1.0f))),qrmxd(1));
// WRF:17603
qrfrzf(1) = qrfrz(1);
// WRF:17606
} else if((ipconc>=3)) {
// WRF:17610
frach = 1.0e0;
// WRF:17613
if(((ibiggopt==2)&&(imurain==1))) {
// WRF:17616
volt = (gfk_exp((16.2f+(1.0f*temcg(1))))*1.0e-6f);
// WRF:17619
dbigg = lake_pow(((lake_div(6.f,pi)*volt)),(lake_div(1.f,3.f)));
// WRF:17622
if((dbigg<8.e-3f)) {
// WRF:17624
ratio = lake_min(maxratiolu,lake_div(dbigg,xdia(1,lr,1)));
// WRF:17626
i = lake_min(nqiacrratio,((int)((ratio*dqiacrratioinv))));
// WRF:17627
if(alp0flag) {
// WRF:17628
j = ((int)((lake_max(0.0f,lake_min(15.f,alpha(1,lr)))*dqiacralphainv)));
// WRF:17629
} else {
// WRF:17630
j = ((int)((lake_max(minalphalu,lake_min(maxalphalu,alpha(1,lr)))*dqiacralphainv)));
// WRF:17631
}
// WRF:17632
delx = (ratio-(float(i)*dqiacrratio));
// WRF:17633
dely = (alpha(1,lr)-(float(j)*dqiacralpha));
// WRF:17634
ip1 = lake_min((i+1),nqiacrratio);
// WRF:17635
jp1 = lake_min((j+1),nqiacralpha);
// WRF:17638
tmp1 = (ciacrratio(i,j)+((delx*dqiacrratioinv)*((ciacrratio(ip1,j)-ciacrratio(i,j)))));
// WRF:17639
tmp2 = (ciacrratio(i,jp1)+((delx*dqiacrratioinv)*((ciacrratio(ip1,jp1)-ciacrratio(i,jp1)))));
// WRF:17643
crfrz(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*cx(1,lr))*dtpinv);
// WRF:17644
crfrzf(1) = crfrz(1);
// WRF:17646
tmp1 = (qiacrratio(i,j)+((delx*dqiacrratioinv)*((qiacrratio(ip1,j)-qiacrratio(i,j)))));
// WRF:17647
tmp2 = (qiacrratio(i,jp1)+((delx*dqiacrratioinv)*((qiacrratio(ip1,jp1)-qiacrratio(i,jp1)))));
// WRF:17651
qrfrz(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*qx(1,lr))*dtpinv);
// WRF:17652
qrfrzf(1) = qrfrz(1);
// WRF:17654
if((((qrfrz(1)*dtp)<coeff_qxmin{}(lh))||((crfrz(1)*dtp)<cxmin))) {
// WRF:17656
crfrz(1) = 0.0f;
// WRF:17657
qrfrz(1) = 0.0f;
// WRF:17658
qrfrzf(1) = 0.0f;
// WRF:17660
} else {
// WRF:17663
if(((ipconc>=6)&&(lzr>1))) {
// WRF:17665
tmp1 = (ziacrratio(i,j)+((delx*dqiacrratioinv)*((ziacrratio(ip1,j)-ziacrratio(i,j)))));
// WRF:17666
tmp2 = (ziacrratio(i,jp1)+((delx*dqiacrratioinv)*((ziacrratio(ip1,jp1)-ziacrratio(i,jp1)))));
// WRF:17670
zrfrz(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*zx(1,lr))*dtpinv);
// WRF:17671
}
// WRF:17673
if((((ibiggsmallrain>0)&&(xv(1,lr)<(2.f*coeff_xvmn{}(lr))))&&(((ibiggsnow==1)||(ibiggsnow==3))))) {
// WRF:17676
crfrzf(1) = 0.0f;
// WRF:17677
qrfrzf(1) = 0.0f;
// WRF:17678
crfrzs(1) = crfrz(1);
// WRF:17679
qrfrzs(1) = qrfrz(1);
// WRF:17681
if(((ipconc>=6)&&(lzr>1))) {
// WRF:17682
zrfrzs(1) = zrfrz(1);
// WRF:17683
zrfrzf(1) = 0.f;
// WRF:17684
}
// WRF:17685
} else if(((dbigg<lake_max(biggsnowdiam,lake_max(dfrz,dhmn)))&&(((ibiggsnow==1)||(ibiggsnow==3))))) {
// WRF:17688
crfrzs(1) = crfrz(1);
// WRF:17689
qrfrzs(1) = qrfrz(1);
// WRF:17691
if(((ibiggsmallrain>0)&&(xv(1,lr)<(1.2f*coeff_xvmn{}(lr))))) {
// WRF:17693
crfrzf(1) = 0.0f;
// WRF:17694
qrfrzf(1) = 0.0f;
// WRF:17696
if(((ipconc>=6)&&(lzr>1))) {
// WRF:17697
zrfrzs(1) = zrfrz(1);
// WRF:17698
zrfrzf(1) = 0.f;
// WRF:17699
}
// WRF:17700
} else {
// WRF:17703
ratio = lake_min(maxratiolu,lake_div(lake_max(dfrz,dhmn),xdia(1,lr,1)));
// WRF:17705
i = lake_min(nqiacrratio,((int)((ratio*dqiacrratioinv))));
// WRF:17708
if(alp0flag) {
// WRF:17709
j = ((int)((lake_max(0.0f,lake_min(15.f,alpha(1,lr)))*dqiacralphainv)));
// WRF:17710
} else {
// WRF:17711
j = ((int)((lake_max(minalphalu,lake_min(maxalphalu,alpha(1,lr)))*dqiacralphainv)));
// WRF:17712
}
// WRF:17713
delx = (ratio-(float(i)*dqiacrratio));
// WRF:17714
dely = (alpha(1,lr)-(float(j)*dqiacralpha));
// WRF:17715
ip1 = lake_min((i+1),nqiacrratio);
// WRF:17716
jp1 = lake_min((j+1),nqiacralpha);
// WRF:17719
tmp1 = (ciacrratio(i,j)+((delx*dqiacrratioinv)*((ciacrratio(ip1,j)-ciacrratio(i,j)))));
// WRF:17720
tmp2 = (ciacrratio(i,jp1)+((delx*dqiacrratioinv)*((ciacrratio(ip1,jp1)-ciacrratio(i,jp1)))));
// WRF:17725
crfrzf(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*cx(1,lr))*dtpinv);
// WRF:17728
tmp1 = (qiacrratio(i,j)+((delx*dqiacrratioinv)*((qiacrratio(ip1,j)-qiacrratio(i,j)))));
// WRF:17729
tmp2 = (qiacrratio(i,jp1)+((delx*dqiacrratioinv)*((qiacrratio(ip1,jp1)-qiacrratio(i,jp1)))));
// WRF:17733
qrfrzf(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*qx(1,lr))*dtpinv);
// WRF:17736
crfrzs(1) = (crfrzs(1)-crfrzf(1));
// WRF:17737
qrfrzs(1) = (qrfrzs(1)-qrfrzf(1));
// WRF:17739
if(((ipconc>=6)&&(lzr>1))) {
// WRF:17740
zrfrzs(1) = zrfrz(1);
// WRF:17742
tmp1 = (ziacrratio(i,j)+((delx*dqiacrratioinv)*((ziacrratio(ip1,j)-ziacrratio(i,j)))));
// WRF:17743
tmp2 = (ziacrratio(i,jp1)+((delx*dqiacrratioinv)*((ziacrratio(ip1,jp1)-ziacrratio(i,jp1)))));
// WRF:17747
zrfrzf(1) = ((((tmp1+((dely*dqiacralphainv)*((tmp2-tmp1)))))*zx(1,lr))*dtpinv);
// WRF:17748
zrfrzs(1) = (zrfrzs(1)-zrfrzf(1));
// WRF:17749
zrfrzf(1) = (lake_pow((lake_div(1000.f,900.f)),2)*zrfrzf(1));
// WRF:17750
}
// WRF:17751
}
// WRF:17752
} else {
// WRF:17753
crfrzs(1) = 0.0f;
// WRF:17754
qrfrzs(1) = 0.0f;
// WRF:17755
zrfrzs(1) = 0.0f;
// WRF:17756
}
// WRF:17758
}
// WRF:17760
if((((qrfrz(1))*dtp)>qx(1,lr))) {
// WRF:17761
fac = lake_div(qx(1,lr),(((qrfrz(1))*dtp)));
// WRF:17762
qrfrz(1) = (fac*qrfrz(1));
// WRF:17763
qrfrzs(1) = (fac*qrfrzs(1));
// WRF:17764
qrfrzf(1) = (fac*qrfrzf(1));
// WRF:17765
crfrz(1) = (fac*crfrz(1));
// WRF:17766
crfrzs(1) = (fac*crfrzs(1));
// WRF:17767
crfrzf(1) = (fac*crfrzf(1));
// WRF:17768
if(((ipconc>=6)&&(lzr>1))) {
// WRF:17769
zrfrz(1) = (fac*zrfrz(1));
// WRF:17770
zrfrzf(1) = (fac*zrfrzf(1));
// WRF:17771
}
// WRF:17772
}
// WRF:17774
}
// WRF:17789
} else if((ibiggopt==1)) {
// WRF:17791
tmp = (((xv(1,lr)*brz)*cx(1,lr))*((gfk_exp(lake_max(((-arz)*temcg(1)),0.0f))-1.0f)));
// WRF:17792
if((false&&(tmp>cxmxd(1,lr)))) {
// WRF:17796
crfrz(1) = cxmxd(1,lr);
// WRF:17797
qrfrz(1) = qxmxd(1,lr);
// WRF:17799
} else {
// WRF:17800
crfrz(1) = tmp;
// WRF:17807
if((lzr<1)) {
// WRF:17808
if((imurain==3)) {
// WRF:17809
bfnu = bfnu0;
// WRF:17810
} else {
// WRF:17811
bfnu = bfnu1;
// WRF:17812
}
// WRF:17813
} else {
// WRF:17815
if((imurain==3)) {
// WRF:17816
bfnu = lake_div(((alpha(1,lr)+2.0f)),((alpha(1,lr)+1.f)));
// WRF:17817
} else {
// WRF:17819
bfnu = lake_div(((((4.f+alpha(1,lr)))*((5.f+alpha(1,lr))))*((6.f+alpha(1,lr)))),(((((1.f+alpha(1,lr)))*((2.f+alpha(1,lr))))*((3.f+alpha(1,lr))))));
// WRF:17822
}
// WRF:17823
}
// WRF:17824
qrfrz(1) = (((bfnu*xmas(1,lr))*rhoinv(1))*crfrz(1));
// WRF:17826
qrfrz(1) = lake_min(qrfrz(1),((1.f*qx(1,lr))*dtpinv));
// WRF:17827
crfrz(1) = lake_min(crfrz(1),((1.f*cx(1,lr))*dtpinv));
// WRF:17828
qrfrz(1) = lake_min(qrfrz(1),qx(1,lr));
// WRF:17829
qrfrzf(1) = qrfrz(1);
// WRF:17830
}
// WRF:17835
if((crfrz(1)>coeff_qxmin{}(lh))) {
// WRF:17840
if(((((ibiggsnow==1)||(ibiggsnow==3)))&&(ibiggopt!=2))) {
// WRF:17841
xvfrz = lake_div((rho0(1)*qrfrz(1)),((crfrz(1)*900.f)));
// WRF:17842
frach = (0.5f*((1.f+tanhf((0.2e12f*((xvfrz-(1.15f*coeff_xvmn{}(lh)))))))));
// WRF:17844
qrfrzs(1) = (((1.f-frach))*qrfrz(1));
// WRF:17845
crfrzs(1) = (((1.f-frach))*crfrz(1));
// WRF:17848
}
// WRF:17850
if(((ipconc>=14)&&(lake_div(((1.e-3f*rho0(1))*qrfrz(1)),crfrz(1))<coeff_xvmn{}(lh)))) {
// WRF:17851
qrfrzs(1) = qrfrz(1);
// WRF:17852
crfrzs(1) = crfrz(1);
// WRF:17853
} else {
// WRF:17856
qrfrzf(1) = (frach*qrfrz(1));
// WRF:17858
if((ibfr<=1)) {
// WRF:17859
crfrzf(1) = (frach*lake_min(crfrz(1),(lake_div(qrfrz(1),((((bfnu*1.0f)*vr1mm)*1000.0f)))*rho0(1))));
// WRF:17860
} else if((ibfr==5)) {
// WRF:17861
crfrzf(1) = ((frach*lake_min(crfrz(1),(lake_div(qrfrz(1),(((bfnu*vfrz)*1000.0f)))*rho0(1))))*rzxh(1));
// WRF:17862
} else if((ibfr==2)) {
// WRF:17863
crfrzf(1) = (frach*lake_min(crfrz(1),(lake_div(qrfrz(1),(((bfnu*vfrz)*1000.0f)))*rho0(1))));
// WRF:17864
} else if((ibfr==6)) {
// WRF:17865
crfrzf(1) = (frach*lake_max(crfrz(1),(lake_div(qrfrz(1),((((bfnu*9.f)*xv(1,lr))*1000.0f)))*rho0(1))));
// WRF:17866
} else {
// WRF:17867
crfrzf(1) = (frach*crfrz(1));
// WRF:17868
}
// WRF:17874
}
// WRF:17876
} else {
// WRF:17877
crfrz(1) = 0.0f;
// WRF:17878
qrfrz(1) = 0.0f;
// WRF:17879
}
// WRF:17881
}
// WRF:17883
if((coeff_lvol{}(lh)>1)) {
// WRF:17884
vrfrzf(1) = lake_div((rho0(1)*qrfrzf(1)),rhofrz);
// WRF:17885
}
// WRF:17888
if((nsplinter!=0)) {
// WRF:17889
if((nsplinter>=1000)) {
// WRF:17892
tmp = 0;
// WRF:17893
if((((qrfrz(1)*dtp)>coeff_qxmin{}(lh))&&(crfrz(1)>1.e-3f))) {
// WRF:17894
tmpdiam = (1.e6f*lake_pow((lake_div((6.f*qrfrz(1)),(((1000.f*pi)*crfrz(1))))),(lake_div(1.f,3.f))));
// WRF:17895
tmp = ((lawson_splinter_fac*lake_pow(tmpdiam,4))*crfrz(1));
// WRF:17896
}
// WRF:17897
} else if((nsplinter>0)) {
// WRF:17898
tmp = (nsplinter*crfrz(1));
// WRF:17899
} else {
// WRF:17900
tmp = ((-nsplinter)*crfrzf(1));
// WRF:17901
}
// WRF:17902
csplinter2(1) = tmp;
// WRF:17903
qsplinter2(1) = lake_min((0.1f*qrfrz(1)),lake_div((tmp*splintermass),rho0(1)));
// WRF:17907
}
// WRF:17918
}
// WRF:17920
} else {
// WRF:17922
}
// WRF:17923
}
// WRF:17925
}
// WRF:17930
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:17931
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:17932
qwfrz(1) = 0.0f;
// WRF:17933
cwfrz(1) = 0.0f;
// WRF:17934
qwfrzc(1) = 0.0f;
// WRF:17935
cwfrzc(1) = 0.0f;
// WRF:17936
qwfrzp(1) = 0.0f;
// WRF:17937
cwfrzp(1) = 0.0f;
// WRF:17938
if((((ibfc>=1)&&(ibfc!=3))&&(temg(1)<268.15f))) {
// WRF:17941
if(((qx(1,lc)>coeff_qxmin{}(lc))&&(cx(1,lc)>cxmin))) {
// WRF:17942
if((ipconc<2)) {
// WRF:17943
qwfrz(1) = ((((lake_div(((2.0f)*(brz)),((xdn(1,lc)*cx(1,lc)))))*((gfk_exp(lake_max(((-arz)*temcg(1)),0.0f))-1.0f)))*rho0(1))*(lake_pow(qx(1,lc),2)));
// WRF:17946
qwfrz(1) = lake_max(qwfrz(1),0.0f);
// WRF:17947
qwfrz(1) = lake_min(qwfrz(1),qcmxd(1));
// WRF:17948
cwfrz(1) = lake_div((qwfrz(1)*rho0(1)),xmas(1,li));
// WRF:17949
} else if((ipconc>=2)) {
// WRF:17950
if((xdia(1,lc,3)>0.e-6f)) {
// WRF:17951
volt = (gfk_exp((16.2f+(1.0f*temcg(1))))*1.0e-6f);
// WRF:17956
if((alpha(1,lc)==0.0f)) {
// WRF:17957
cwfrz(1) = ((cx(1,lc)*gfk_exp(lake_div((-volt),xv(1,lc))))*dtpinv);
// WRF:17961
qwfrz(1) = (((cwfrz(1)*coeff_xdn0{}(lc))*rhoinv(1))*((volt+xv(1,lc))));
// WRF:17962
} else {
// WRF:17963
ratio = lake_div((((1.f+alpha(1,lc)))*volt),xv(1,lc));
// WRF:17965
if((false&&usegamxinfcnu)) {
// WRF:17966
i = ((int)round((dgami*((1.f+alpha(1,lc))))));
// WRF:17967
gcnup1 = gmoi(i);
// WRF:17968
i = ((int)round((dgami*((2.f+alpha(1,lc))))));
// WRF:17969
gcnup2 = gmoi(i);
// WRF:17971
cwfrz(1) = lake_div((cx(1,lc)*gamxinf((1.f+alpha(1,lc)),ratio)),((dtp*gcnup1)));
// WRF:17973
qwfrz(1) = lake_div(((((cx(1,lc)*coeff_xdn0{}(lc))*xv(1,lc))*rhoinv(1))*gamxinf((2.f+alpha(1,lc)),ratio)),((dtp*gcnup2)));
// WRF:17975
} else {
// WRF:17977
ratio = lake_min(maxratiolu,ratio);
// WRF:17981
tmp = gaminterp(ratio,alpha(1,lc),1,1);
// WRF:17983
cwfrz(1) = ((cx(1,lc)*tmp)*dtpinv);
// WRF:17985
tmp = gaminterp(ratio,alpha(1,lc),12,1);
// WRF:17987
qwfrz(1) = (((((cx(1,lc)*coeff_xdn0{}(lc))*xv(1,lc))*rhoinv(1))*dtpinv)*tmp);
// WRF:17989
}
// WRF:17991
}
// WRF:17993
}
// WRF:17994
}
// WRF:17995
if((temg(1)>268.15f)) {
// WRF:17996
qwfrz(1) = 0.0f;
// WRF:17997
cwfrz(1) = 0.0f;
// WRF:17998
}
// WRF:17999
}
// WRF:18000
}
// WRF:18002
if((xplate(1)==1)) {
// WRF:18003
qwfrzp(1) = qwfrz(1);
// WRF:18004
cwfrzp(1) = cwfrz(1);
// WRF:18005
}
// WRF:18007
if((xcolmn(1)==1)) {
// WRF:18008
qwfrzc(1) = qwfrz(1);
// WRF:18009
cwfrzc(1) = cwfrz(1);
// WRF:18010
}
// WRF:18016
}
// WRF:18022
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:18023
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18025
ccia(1) = 0.0f;
// WRF:18027
cwctfz(1) = 0.0f;
// WRF:18028
qwctfz(1) = 0.0f;
// WRF:18029
ctfzbd(1) = 0.0f;
// WRF:18030
ctfzth(1) = 0.0f;
// WRF:18031
ctfzdi(1) = 0.0f;
// WRF:18033
cwctfzc(1) = 0.0f;
// WRF:18034
qwctfzc(1) = 0.0f;
// WRF:18035
cwctfzp(1) = 0.0f;
// WRF:18036
qwctfzp(1) = 0.0f;
// WRF:18037
if((icfn>=1)) {
// WRF:18039
if(((temg(1)<271.15f)&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:18043
if((icfn>=2)) {
// WRF:18044
ccia(1) = gfk_exp((4.11f-((0.262f)*temcg(1))));
// WRF:18050
knud(1) = lake_div((2.28e-5f*temg(1)),((pres(1)*raero)));
// WRF:18051
knuda(1) = (1.257f+(0.4f*gfk_exp(lake_div((-1.1f),knud(1)))));
// WRF:18052
gtp(1) = lake_div(1.f,((fai(1)+fbi(1))));
// WRF:18053
dfar(1) = lake_div(((kb*temg(1))*((1.f+(knuda(1)*knud(1))))),((((6.f*pi)*fadvisc(1))*raero)));
// WRF:18054
fn1(1) = ((((2.f*pi)*xdia(1,lc,1))*cx(1,lc))*ccia(1));
// WRF:18055
fn2(1) = lake_div((((-gtp(1))*((ssw(1)-1.f)))*felv(1)),pres(1));
// WRF:18056
fnft(1) = lake_div(((0.4f*(((1.f+(1.45f*knud(1)))+((0.4f*knud(1))*gfk_exp(lake_div((-1.f),knud(1)))))))*((ftka(1)+((2.5f*knud(1))*kaero)))),((((1.f+(3.f*knud(1))))*((((2*ftka(1))+((5.f*knud(1))*kaero))+kaero)))));
// WRF:18061
ctfzbd(1) = (fn1(1)*dfar(1));
// WRF:18064
ctfzth(1) = lake_div(((fn1(1)*fn2(1))*fnft(1)),rho0(1));
// WRF:18067
ctfzdi(1) = lake_div((((fn1(1)*fn2(1))*rw)*temg(1)),((felv(1)*rho0(1))));
// WRF:18069
cwctfz(1) = lake_max(((ctfzbd(1)+ctfzth(1))+ctfzdi(1)),0.f);
// WRF:18079
} else if((icfn==1)) {
// WRF:18080
if((wvel(1)<(-0.05f))) {
// WRF:18081
cwctfz(1) = (cfnfac*gfk_exp((((-2.80f))-((0.262f)*temcg(1)))));
// WRF:18082
cwctfz(1) = lake_min(((1.0e3f)*cwctfz(1)),ccmxd(1));
// WRF:18083
}
// WRF:18084
}
// WRF:18086
if((ipconc>=2)) {
// WRF:18087
cwctfz(1) = lake_min((cwctfz(1)*dtpinv),ccmxd(1));
// WRF:18088
qwctfz(1) = lake_div((xmas(1,lc)*cwctfz(1)),rho0(1));
// WRF:18089
} else {
// WRF:18090
qwctfz(1) = lake_div(((cimasn)*cwctfz(1)),((dtp*rho0(1))));
// WRF:18091
qwctfz(1) = lake_max(qwctfz(1),0.0f);
// WRF:18092
qwctfz(1) = lake_min(qwctfz(1),qcmxd(1));
// WRF:18093
}
// WRF:18096
if((xplate(1)==1)) {
// WRF:18097
qwctfzp(1) = qwctfz(1);
// WRF:18098
cwctfzp(1) = cwctfz(1);
// WRF:18099
}
// WRF:18101
if((xcolmn(1)==1)) {
// WRF:18102
qwctfzc(1) = qwctfz(1);
// WRF:18103
cwctfzc(1) = cwctfz(1);
// WRF:18104
}
// WRF:18114
}
// WRF:18116
}
// WRF:18118
}
// WRF:18124
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:18125
dthr = 300.0f;
// WRF:18126
hrifac = ((1.e-3f)*(((0.044f)*(lake_pow(0.01f,3)))));
// WRF:18127
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18128
ciihr(1) = 0.0f;
// WRF:18129
qiihr(1) = 0.0f;
// WRF:18130
cicichr(1) = 0.0f;
// WRF:18131
qicichr(1) = 0.0f;
// WRF:18132
cipiphr(1) = 0.0f;
// WRF:18133
qipiphr(1) = 0.0f;
// WRF:18134
if((ihrn>=1)) {
// WRF:18135
if((qx(1,lc)>coeff_qxmin{}(lc))) {
// WRF:18136
if((temg(1)<273.15f)) {
// WRF:18147
if((gfk_log(lake_div((cx(1,lc)*(1.e-6f)),(3.0f)))>0.0f)) {
// WRF:18148
ciihr(1) = ((lake_div((1.69e17f),dthr))*lake_pow((lake_div((gfk_log(lake_div((cx(1,lc)*(1.e-6f)),(3.0f)))*((((1.e-3f)*rho0(1))*qx(1,lc)))),((cx(1,lc)*(1.e-6f))))),(lake_div(7.f,3.f))));
// WRF:18151
ciihr(1) = (ciihr(1)*(1.0e6f));
// WRF:18152
qiihr(1) = lake_div((hrifac*ciihr(1)),rho0(1));
// WRF:18153
qiihr(1) = lake_max(qiihr(1),0.0f);
// WRF:18154
qiihr(1) = lake_min(qiihr(1),qcmxd(1));
// WRF:18155
}
// WRF:18157
if((xplate(1)==1)) {
// WRF:18158
qipiphr(1) = qiihr(1);
// WRF:18159
cipiphr(1) = ciihr(1);
// WRF:18160
}
// WRF:18162
if((xcolmn(1)==1)) {
// WRF:18163
qicichr(1) = qiihr(1);
// WRF:18164
cicichr(1) = ciihr(1);
// WRF:18165
}
// WRF:18170
}
// WRF:18171
}
// WRF:18172
}
// WRF:18173
}
// WRF:18198
hdia0 = 300.0e-6f;
// WRF:18199
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18200
qscnvi(1) = 0.0f;
// WRF:18201
cscnvi(1) = 0.0f;
// WRF:18202
cscnvis(1) = 0.0f;
// WRF:18205
if(((temg(1)<tfr)&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:18206
if(((ipconc>=4)&&false)) {
// WRF:18207
if(((cx(1,li)>10.f)&&(xdia(1,li,1)>50.e-6f))) {
// WRF:18208
cirdiatmp = lake_pow((lake_div((qx(1,li)*rho0(1)),(((pi*xdn(1,li))*cx(1,li))))),(lake_div(1.f,3.f)));
// WRF:18211
if((cirdiatmp>100.e-6f)) {
// WRF:18212
qscnvi(1) = (((lake_div((((pi*xdn(1,li))*cx(1,li))),(((6.0f*rho0(1))*dtp))))*gfk_exp(lake_div((-hdia0),cirdiatmp)))*(((((lake_pow(hdia0,3))+((3.0f*(lake_pow(hdia0,2)))*cirdiatmp))+((6.0f*(hdia0))*(lake_pow(cirdiatmp,2))))+(6.0f*(lake_pow(cirdiatmp,3))))));
// WRF:18217
qscnvi(1) = lake_min(qscnvi(1),qimxd(1));
// WRF:18219
if((ipconc>=4)) {
// WRF:18220
cscnvi(1) = lake_min(cimxd(1),(cx(1,li)*gfk_exp(lake_div((-hdia0),cirdiatmp))));
// WRF:18221
}
// WRF:18222
}
// WRF:18223
}
// WRF:18225
} else if((ipconc<4)) {
// WRF:18227
qscnvi(1) = ((0.001f*eii(1))*lake_max(((qx(1,li)-1.e-3f)),0.0f));
// WRF:18228
qscnvi(1) = lake_min(qscnvi(1),qxmxd(1,li));
// WRF:18229
cscnvi(1) = lake_div((qscnvi(1)*rho0(1)),xmas(1,li));
// WRF:18230
cscnvis(1) = (0.5f*cscnvi(1));
// WRF:18232
}
// WRF:18233
}
// WRF:18235
}
// WRF:18242
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18243
fvent(1) = ((lake_pow(fschm(1),(lake_div(1.f,3.f))))*(lake_pow(fakvisc(1),((-0.5f)))));
// WRF:18244
}
// WRF:18247
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:18249
civenta = 1.258e4f;
// WRF:18250
civentb = 2.331f;
// WRF:18251
civentc = 5.662e4f;
// WRF:18252
civentd = 2.373f;
// WRF:18253
civente = 0.8241f;
// WRF:18254
civentf = (-0.042f);
// WRF:18255
civentg = 1.70f;
// WRF:18257
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18258
if((((icond==1)||(temg(1)<=tfrh))||(((qx(1,lr)<=coeff_qxmin{}(lr))&&(qx(1,lc)<=coeff_qxmin{}(lc)))))) {
// WRF:18260
if((qx(1,li)>coeff_qxmin{}(li))) {
// WRF:18261
cireyn = lake_div((((civenta*lake_pow(xdia(1,li,1),civentb))+(civentc*lake_pow(xdia(1,li,1),civentd)))),(((civente*lake_pow(xdia(1,li,1),civentf))+civentg)));
// WRF:18266
xcivent = ((lake_pow(fschm(1),(lake_div(1.f,3.f))))*(lake_pow((lake_div(cireyn,fakvisc(1))),0.5f)));
// WRF:18267
if((xcivent<1.0f)) {
// WRF:18268
civent(1) = (1.0f+(0.14f*lake_pow(xcivent,2)));
// WRF:18269
}
// WRF:18270
if((xcivent>=1.0f)) {
// WRF:18271
civent(1) = (0.86f+(0.28f*xcivent));
// WRF:18272
}
// WRF:18273
} else {
// WRF:18274
civent(1) = 0.0f;
// WRF:18275
}
// WRF:18278
}
// WRF:18279
}
// WRF:18283
igmrwa = (100.0f*2.0f);
// WRF:18284
igmrwb = (100.f*(lake_div(((5.0f+br)),2.0f)));
// WRF:18285
rwventa = ((0.78f)*gmoi(igmrwa));
// WRF:18286
rwventb = ((0.308f)*gmoi(igmrwb));
// WRF:18287
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18288
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:18289
if((ipconc>=3)) {
// WRF:18290
if((imurain==3)) {
// WRF:18291
if((izwisventr==1)) {
// WRF:18292
rwvent(1) = (ventrx(1)*((1.6f+(124.9f*lake_pow((((1.e-3f*rho0(1))*qx(1,lr))),.2046f)))));
// WRF:18293
} else {
// WRF:18295
rwvent(1) = (((0.78f*ventrx(1))+((((0.308f*ventrxn(1))*fvent(1))*sqrtf(((ar*rhovt(1)))))*(lake_pow(xdia(1,lr,1),(lake_div(((1.0f+br)),2.0f)))))));
// WRF:18299
}
// WRF:18301
} else {
// WRF:18308
if((iferwisventr==1)) {
// WRF:18312
alpr = lake_min(alpharmax,alpha(1,lr));
// WRF:18314
x = (1.f+alpha(1,lr));
// WRF:18316
if(((ipconc>=6)&&(lzr>1))) {
// WRF:18317
tmp = (1.f+alpr);
// WRF:18318
i = ((int)((dgami*(tmp))));
// WRF:18319
del = (tmp-(dgam*i));
// WRF:18320
g1palp = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18322
tmp = ((2.5f+alpha(1,lr))+(0.5f*coeff_bx{}(lr)));
// WRF:18323
i = ((int)((dgami*(tmp))));
// WRF:18324
del = (tmp-(dgam*i));
// WRF:18325
y = lake_div(((gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami))),g1palp);
// WRF:18326
} else {
// WRF:18327
y = ventrxn(1);
// WRF:18328
}
// WRF:18332
vent1 = lake_pow(((double)(xdia(1,lr,1))),((0.5f+(0.5f*coeff_bx{}(lr)))));
// WRF:18333
vent2 = lake_pow(((double)((1.f+((0.5f*coeff_fx{}(lr))*xdia(1,lr,1))))),((double)(((2.5f+alpr)+(0.5f*coeff_bx{}(lr))))));
// WRF:18336
rwvent(1) = ((0.78f*x)+((((0.308f*fvent(1))*y)*sqrtf((coeff_ax{}(lr)*rhovt(1))))*(lake_div(vent1,vent2))));
// WRF:18341
rwventz(1) = 0.0f;
// WRF:18349
} else if((iferwisventr==2)) {
// WRF:18352
x = (1.f+alpha(1,lr));
// WRF:18354
rwvent(1) = (((0.78f*x)+((((0.308f*ventrxn(1))*fvent(1))*sqrtf(((ar*rhovt(1)))))*(lake_pow(xdia(1,lr,1),(lake_div(((1.0f+br)),2.0f)))))));
// WRF:18360
if((ipconc>=7)) {
// WRF:18361
alpr = lake_min(alpharmax,alpha(1,lr));
// WRF:18363
tmp = ((alpr+5.5f)+lake_div(br,2.f));
// WRF:18364
i = ((int)((dgami*(tmp))));
// WRF:18365
del = (tmp-(dgam*i));
// WRF:18366
y = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18370
rwventz(1) = (((((0.78f*((4.f+alpr)))*((3.f+alpr)))*((2.f+alpr)))*((1.f+alpr)))+((((0.308f*fvent(1))*sqrtf((coeff_ax{}(lr)*rhovt(1))))*(lake_div(y,gf1palp(1))))*(lake_pow(xdia(1,lr,1),(lake_div(((1.0f+br)),2.0f))))));
// WRF:18375
}
// WRF:18378
}
// WRF:18380
}
// WRF:18381
} else {
// WRF:18382
rwvent(1) = ((rwventa+(((rwventb*fvent(1))*sqrtf(((ar*rhovt(1)))))*(lake_pow(xdia(1,lr,1),(lake_div(((1.0f+br)),2.0f)))))));
// WRF:18386
}
// WRF:18387
} else {
// WRF:18388
rwvent(1) = 0.0f;
// WRF:18389
}
// WRF:18390
}
// WRF:18392
igmswa = (100.0f*2.0f);
// WRF:18393
igmswb = (100.f*(lake_div(((5.0f+ds)),2.0f)));
// WRF:18394
swventa = ((0.78f)*gmoi(igmswa));
// WRF:18395
swventb = ((0.308f)*gmoi(igmswb));
// WRF:18396
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18397
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:18398
if((ipconc>=4)) {
// WRF:18399
swvent(1) = (0.65f+((0.44f*fvent(1))*sqrtf((vtxbar(1,ls,1)*xdia(1,ls,1)))));
// WRF:18400
} else {
// WRF:18402
swvent(1) = ((swventa+(((swventb*fvent(1))*sqrtf(((cs*rhovt(1)))))*(lake_pow(xdia(1,ls,1),(lake_div(((1.0f+ds)),2.0f)))))));
// WRF:18406
}
// WRF:18407
} else {
// WRF:18408
swvent(1) = 0.0f;
// WRF:18409
}
// WRF:18410
}
// WRF:18414
igmhwa = (100.0f*2.0f);
// WRF:18415
igmhwb = (100.0f*2.75f);
// WRF:18416
hwventa = ((0.78f)*gmoi(igmhwa));
// WRF:18417
hwventb = ((0.308f)*gmoi(igmhwb));
// WRF:18419
for(int _s0=1;_s0<=ngs;++_s0) {hwvent(_s0) = 0.0f;}
// WRF:18420
for(int _s0=1;_s0<=ngs;++_s0) {hwventy(_s0) = 0.0f;}
// WRF:18422
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18423
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:18424
hwventc = lake_pow((lake_div((4.0f*gr),((3.0f*cdxgs(1,lh))))),(0.25f));
// WRF:18425
if((false||(alpha(1,lh)==0.0f))) {
// WRF:18426
hwvent(1) = ((hwventa+((((hwventb*hwventc)*fvent(1))*(lake_pow((lake_div(xdn(1,lh),rho0(1))),(0.25f))))*(lake_pow(xdia(1,lh,1),(0.75f))))));
// WRF:18430
} else {
// WRF:18439
x = (1.f+alpha(1,lh));
// WRF:18441
tmp = (1+alpha(1,lh));
// WRF:18442
i = ((int)((dgami*(tmp))));
// WRF:18443
del = (tmp-(dgam*i));
// WRF:18444
g1palp = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18446
tmp = ((2.5f+alpha(1,lh))+(0.5f*bxx(1,lh)));
// WRF:18447
i = ((int)((dgami*(tmp))));
// WRF:18448
del = (tmp-(dgam*i));
// WRF:18449
y = lake_div(((gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami))),g1palp);
// WRF:18452
hwventy(1) = (((0.308f*fvent(1))*(lake_pow(xdia(1,lh,1),((0.5f+(0.5f*bxx(1,lh)))))))*sqrtf((axx(1,lh)*rhovt(1))));
// WRF:18453
hwvent(1) = (((0.78f*x)+(y*hwventy(1))));
// WRF:18458
}
// WRF:18459
} else {
// WRF:18460
hwvent(1) = 0.0f;
// WRF:18461
hwventy(1) = 0.0f;
// WRF:18462
}
// WRF:18463
}
// WRF:18466
for(int _s0=1;_s0<=ngs;++_s0) {hlvent(_s0) = 0.0f;}
// WRF:18467
for(int _s0=1;_s0<=ngs;++_s0) {hlventy(_s0) = 0.0f;}
// WRF:18469
if((lhl>1)) {
// WRF:18470
igmhwa = (100.0f*2.0f);
// WRF:18471
igmhwb = (100.0f*2.75f);
// WRF:18472
hwventa = ((0.78f)*gmoi(igmhwa));
// WRF:18473
hwventb = ((0.308f)*gmoi(igmhwb));
// WRF:18475
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18476
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:18477
hwventc = lake_pow((lake_div((4.0f*gr),((3.0f*cdxgs(1,lhl))))),(0.25f));
// WRF:18479
if((false||(alpha(1,lhl)==0.0f))) {
// WRF:18480
hlvent(1) = ((hwventa+((((hwventb*hwventc)*fvent(1))*(lake_pow((lake_div(xdn(1,lhl),rho0(1))),(0.25f))))*(lake_pow(xdia(1,lhl,1),(0.75f))))));
// WRF:18484
} else {
// WRF:18494
x = (1.f+alpha(1,lhl));
// WRF:18496
tmp = (1+alpha(1,lhl));
// WRF:18497
i = ((int)((dgami*(tmp))));
// WRF:18498
del = (tmp-(dgam*i));
// WRF:18499
g1palp = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18501
tmp = ((2.5f+alpha(1,lhl))+(0.5f*bxx(1,lhl)));
// WRF:18502
i = ((int)((dgami*(tmp))));
// WRF:18503
del = (tmp-(dgam*i));
// WRF:18504
y = lake_div(((gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami))),g1palp);
// WRF:18506
hlventy(1) = (((0.308f*fvent(1))*(lake_pow(xdia(1,lhl,1),((0.5f+(0.5f*bxx(1,lhl)))))))*sqrtf((axx(1,lhl)*rhovt(1))));
// WRF:18508
hlvent(1) = ((0.78f*x)+(y*hlventy(1)));
// WRF:18513
}
// WRF:18514
}
// WRF:18515
}
// WRF:18516
}
// WRF:18523
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18524
fwet1(1) = lake_div((((2.0f*pi))*(((((felv(1)*fwvdf(1))*rho0(1))*((qss0(1)-qx(1,lv))))-(ftka(1)*temcg(1))))),((rho0(1)*((felf(1)+(fcw(1)*temcg(1)))))));
// WRF:18529
fwet2(1) = ((1.0f)-lake_div((fci(1)*temcg(1)),((felf(1)+(fcw(1)*temcg(1))))));
// WRF:18532
}
// WRF:18536
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18537
fmlt1(1) = lake_div((((2.0f*pi))*((((felv(1)*fwvdf(1))*((qss0(1)-qx(1,lv))))-lake_div((ftka(1)*temcg(1)),rho0(1))))),(felf(1)));
// WRF:18541
fmlt2(1) = lake_div(((-fcw(1))*temcg(1)),felf(1));
// WRF:18542
fmlt1e(1) = lake_div((((2.0f*pi))*(((felv(1)*fwvdf(1))*((qss0(1)-qx(1,lv)))))),(felf(1)));
// WRF:18544
}
// WRF:18548
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18549
fvds(1) = (((lake_div((4.0f*pi),rho0(1)))*((ssi(1)-1.0f)))*(lake_div(1.0f,((fai(1)+fbi(1))))));
// WRF:18552
}
// WRF:18553
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18554
fvce(1) = (((lake_div((4.0f*pi),rho0(1)))*((ssw(1)-1.0f)))*(lake_div(1.0f,((fav(1)+fbv(1))))));
// WRF:18557
}
// WRF:18562
for(int _s0=1;_s0<=ngs;++_s0) {qsmlr(_s0) = 0.0f;}
// WRF:18563
for(int _s0=1;_s0<=ngs;++_s0) {qimlr(_s0) = 0.0f;}
// WRF:18564
for(int _s0=1;_s0<=ngs;++_s0) {qhmlr(_s0) = 0.0f;}
// WRF:18565
for(int _s0=1;_s0<=ngs;++_s0) {qhlmlr(_s0) = 0.0f;}
// WRF:18566
if((lhwlg>1)) {
// WRF:18567
for(int _s0=1;_s0<=ngs;++_s0) {qhmlrlg(_s0) = 0.0f;}
// WRF:18568
for(int _s0=1;_s0<=ngs;++_s0) {qhlmlrlg(_s0) = 0.0f;}
// WRF:18569
}
// WRF:18570
for(int _s0=1;_s0<=ngs;++_s0) {qhfzh(_s0) = 0.0f;}
// WRF:18571
for(int _s0=1;_s0<=ngs;++_s0) {qffzf(_s0) = 0.0f;}
// WRF:18572
for(int _s0=1;_s0<=ngs;++_s0) {qhlfzhl(_s0) = 0.0f;}
// WRF:18573
for(int _s0=1;_s0<=ngs;++_s0) {qhfzhlg(_s0) = 0.0f;}
// WRF:18574
for(int _s0=1;_s0<=ngs;++_s0) {qhlfzhllg(_s0) = 0.0f;}
// WRF:18575
for(int _s0=1;_s0<=ngs;++_s0) {vhfzh(_s0) = 0.0f;}
// WRF:18576
for(int _s0=1;_s0<=ngs;++_s0) {vffzf(_s0) = 0.0f;}
// WRF:18577
for(int _s0=1;_s0<=ngs;++_s0) {vhlfzhl(_s0) = 0.0f;}
// WRF:18578
for(int _s0=1;_s0<=ngs;++_s0) {qsfzs(_s0) = 0.0f;}
// WRF:18580
for(int _s0=1;_s0<=ngs;++_s0) {zhmlr(_s0) = 0.0f;}
// WRF:18581
for(int _s0=1;_s0<=ngs;++_s0) {zhmlrr(_s0) = 0.0f;}
// WRF:18582
for(int _s0=1;_s0<=ngs;++_s0) {zsmlrr(_s0) = 0.0f;}
// WRF:18583
for(int _s0=1;_s0<=ngs;++_s0) {zhshr(_s0) = 0.0f;}
// WRF:18584
for(int _s0=1;_s0<=ngs;++_s0) {zhlmlr(_s0) = 0.0f;}
// WRF:18585
for(int _s0=1;_s0<=ngs;++_s0) {zhlshr(_s0) = 0.0f;}
// WRF:18587
for(int _s0=1;_s0<=ngs;++_s0) {zhshrr(_s0) = 0.0f;}
// WRF:18588
for(int _s0=1;_s0<=ngs;++_s0) {zhlmlrr(_s0) = 0.0f;}
// WRF:18589
for(int _s0=1;_s0<=ngs;++_s0) {zhlshrr(_s0) = 0.0f;}
// WRF:18591
for(int _s0=1;_s0<=ngs;++_s0) {csmlr(_s0) = 0.0f;}
// WRF:18592
for(int _s0=1;_s0<=ngs;++_s0) {csmlrr(_s0) = 0.0f;}
// WRF:18593
for(int _s0=1;_s0<=ngs;++_s0) {chmlr(_s0) = 0.0f;}
// WRF:18594
for(int _s0=1;_s0<=ngs;++_s0) {chmlrr(_s0) = 0.0f;}
// WRF:18595
for(int _s0=1;_s0<=ngs;++_s0) {chlmlr(_s0) = 0.0f;}
// WRF:18596
for(int _s0=1;_s0<=ngs;++_s0) {chlfmlr(_s0) = 0.0f;}
// WRF:18601
for(int _s0=1;_s0<=ngs;++_s0) {chlmlrr(_s0) = 0.0f;}
// WRF:18604
if((!mixedphase)) {
// WRF:18605
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18607
if((temg(1)>tfr)) {
// WRF:18609
if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:18610
qsmlr(1) = lake_min((((((c1sw*fmlt1(1))*cx(1,ls))*swvent(1))*xdia(1,ls,1))),0.0f);
// WRF:18614
}
// WRF:18628
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:18630
if(((ibinhmlr==0)||(lzh<1))) {
// WRF:18631
qhmlr(1) = (meltfac*lake_min(((((fmlt1(1)*cx(1,lh))*hwvent(1))*xdia(1,lh,1))+(fmlt2(1)*((qhacrmlr(1)+qhacwmlr(1))))),0.0f));
// WRF:18636
} else if((ibinhmlr==1)) {
// WRF:18638
; /* native diagnostic only */
// WRF:18639
asm("trap;"); return; /* native STOP */
// WRF:18641
} else if(((ibinhmlr==2)||(ibinhmlr==3))) {
// WRF:18643
}
// WRF:18646
if(((((ivhmltsoak>0)&&(qhmlr(1)<0.0f))&&(coeff_lvol{}(lh)>1))&&(xdn(1,lh)<coeff_xdnmx{}(lh)))) {
// WRF:18648
v1 = lake_div((((1.f-lake_div(xdn(1,lh),coeff_xdnmx{}(lh))))*((vx(1,lh)+lake_div((rho0(1)*qhmlr(1)),xdn(1,lh))))),(dtp));
// WRF:18649
v2 = lake_div((((-1.0f)*rho0(1))*qhmlr(1)),coeff_xdnmx{}(lh));
// WRF:18651
vhsoak(1) = lake_min(v1,v2);
// WRF:18653
}
// WRF:18655
}
// WRF:18658
if(((lhl>1)&&(lhlw<1))) {
// WRF:18660
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:18661
if(((ibinhlmlr==0)||(lzhl<1))) {
// WRF:18662
qhlmlr(1) = (meltfac*lake_min(((((fmlt1(1)*cx(1,lhl))*hlvent(1))*xdia(1,lhl,1))+(fmlt2(1)*((qhlacrmlr(1)+qhlacwmlr(1))))),0.0f));
// WRF:18668
} else if((ibinhlmlr==1)) {
// WRF:18673
} else if((ibinhlmlr==(-1))) {
// WRF:18675
}
// WRF:18678
if(((((ivhmltsoak>0)&&(qhlmlr(1)<0.0f))&&(coeff_lvol{}(lhl)>1))&&(xdn(1,lhl)<coeff_xdnmx{}(lhl)))) {
// WRF:18680
v1 = lake_div((((1.f-lake_div(xdn(1,lhl),coeff_xdnmx{}(lhl))))*((vx(1,lhl)+lake_div((rho0(1)*qhlmlr(1)),xdn(1,lhl))))),(dtp));
// WRF:18681
v2 = lake_div((((-1.0f)*rho0(1))*qhlmlr(1)),coeff_xdnmx{}(lhl));
// WRF:18683
vhlsoak(1) = lake_min(v1,v2);
// WRF:18685
}
// WRF:18687
}
// WRF:18688
}
// WRF:18690
}
// WRF:18696
if((!mixedphase)) {qsmlr(1) = lake_max(qsmlr(1),lake_min((-qsmxd(1)),(((-0.7f)*qx(1,ls))*dtpinv)));}
// WRF:18697
if((!mixedphase)) {
// WRF:18698
qhmlr(1) = lake_max(qhmlr(1),lake_min((-qhmxd(1)),(((-0.95f)*qx(1,lh))*dtpinv)));
// WRF:18699
chmlr(1) = lake_max(chmlr(1),lake_min((-chmxd(1)),(((-0.95f)*cx(1,lh))*dtpinv)));
// WRF:18700
}
// WRF:18702
qhmlh(1) = 0.f;
// WRF:18708
if(((lhl>1)&&(lhlw<1))) {
// WRF:18709
qhlmlr(1) = lake_max(qhlmlr(1),lake_min((-qxmxd(1,lhl)),(((-0.95f)*qx(1,lhl))*dtpinv)));
// WRF:18710
chlmlr(1) = lake_max(chlmlr(1),lake_min((-cxmxd(1,lhl)),(((-0.95f)*cx(1,lhl))*dtpinv)));
// WRF:18711
}
// WRF:18714
}
// WRF:18716
}
// WRF:18718
if((ipconc>=1)) {
// WRF:18719
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18720
cimlr(1) = ((lake_div(cx(1,li),((qx(1,li)+1.e-20f))))*qimlr(1));
// WRF:18721
if((!mixedphase)) {
// WRF:18722
if((((xdia(1,ls,1)>1.e-6f)&&((-qsmlr(1))>=(0.5f*coeff_qxmin{}(ls))))&&(ipconc>=4))) {
// WRF:18724
csmlr(1) = ((lake_div(cx(1,ls),(qx(1,ls))))*qsmlr(1));
// WRF:18725
} else if((qx(1,ls)>coeff_qxmin{}(ls))) {
// WRF:18726
csmlr(1) = ((lake_div(cx(1,ls),(qx(1,ls))))*qsmlr(1));
// WRF:18727
}
// WRF:18729
csmlrr(1) = lake_div(csmlr(1),rzxs(1));
// WRF:18730
if((((((-csmlrr(1))*dtp)>cxmin)&&(((-qsmlr(1))*dtp)>coeff_qxmin{}(lr)))&&(snowmeltdia>0.0f))) {
// WRF:18731
rmas = lake_div((rho0(1)*qsmlr(1)),csmlrr(1));
// WRF:18732
if((rmas>snowmeltmass)) {
// WRF:18733
csmlrr(1) = lake_div((rho0(1)*qsmlr(1)),snowmeltmass);
// WRF:18734
}
// WRF:18735
}
// WRF:18743
if(((ibinhmlr==0)||(lzh<1))) {
// WRF:18744
chmlr(1) = ((lake_div(cx(1,lh),((qx(1,lh)+1.e-20f))))*qhmlr(1));
// WRF:18745
if(((imltshddmr==3)&&(qhmlr(1)<(-coeff_qxmin{}(lh))))) {
// WRF:18754
tmp = (1.f+alpha(1,lh));
// WRF:18755
i = ((int)((dgami*(tmp))));
// WRF:18756
del = (tmp-(dgam*i));
// WRF:18757
g1palp = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18759
ratio = lake_min(maxratiolu,lake_div(mltdiam1,xdia(1,lh,1)));
// WRF:18761
x = lake_div(gamxinfdp((2.f+alpha(1,lh)),ratio),g1palp);
// WRF:18762
y = lake_div(gamxinfdp(((2.5f+alpha(1,lh))+(0.5f*bxx(1,lh))),ratio),g1palp);
// WRF:18764
hwvent1 = ((0.78f*x)+(y*hwventy(1)));
// WRF:18766
qhlmlr1 = lake_min((((fmlt1(1)*cx(1,lh))*hwvent1)*xdia(1,lh,1)),0.0f);
// WRF:18768
chmlr(1) = ((lake_div(cx(1,lh),((qx(1,lh)+1.e-20f))))*((qhmlr(1)-qhlmlr1)));
// WRF:18771
}
// WRF:18775
}
// WRF:18779
if(((chmlr(1)<0.0f)&&(((ibinhmlr<1)||(lzh<1))))) {
// WRF:18780
if(((((ipconc>=6)&&(lzr>1))&&(lzh<1))&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:18781
tmp = lake_div(qx(1,lh),cx(1,lh));
// WRF:18782
alp = alpha(1,lh);
// WRF:18783
g1 = g1x(1,lh);
// WRF:18785
zhmlr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*((((2.f*tmp)*qhmlr(1))-(lake_pow(tmp,2)*chmlr(1)))));
// WRF:18787
}
// WRF:18789
if(((ibinhmlr==0)||(lzh<1))) {
// WRF:18790
if((ihmlt==1)) {
// WRF:18791
chmlrr(1) = lake_min(chmlr(1),lake_div((rho0(1)*qhmlr(1)),((xdn(1,lr)*vmlt))));
// WRF:18792
} else if((ihmlt==2)) {
// WRF:18793
if(((xv(1,lh)>0.0f)&&(chmlr(1)<0.0f))) {
// WRF:18796
if((imltshddmr==1)) {
// WRF:18799
tmp = lake_div(((-rho0(1))*qhmlr(1)),(lake_min((xdn(1,lr)*coeff_xvmx{}(lr)),(xdn(1,lh)*xv(1,lh)))));
// WRF:18800
tmp2 = lake_div(((-rho0(1))*qhmlr(1)),((xdn(1,lr)*vr3mm)));
// WRF:18802
chmlrr(1) = (lake_div((tmp*((sheddiam0-xdia(1,lh,3)))),((sheddiam0-sheddiam)))+lake_div((tmp2*((xdia(1,lh,3)-sheddiam))),((sheddiam0-sheddiam))));
// WRF:18803
chmlrr(1) = (-lake_max(tmp,lake_min(tmp2,chmlrr(1))));
// WRF:18804
} else if(((imltshddmr==2)||(imltshddmr==3))) {
// WRF:18807
chmlrr(1) = lake_div((rho0(1)*qhmlr(1)),((xdn(1,lr)*vshdgs(1,lh))));
// WRF:18808
} else {
// WRF:18809
chmlrr(1) = lake_div((rho0(1)*qhmlr(1)),(lake_min((xdn(1,lr)*coeff_xvmx{}(lr)),(xdn(1,lh)*xv(1,lh)))));
// WRF:18810
}
// WRF:18811
} else {
// WRF:18812
chmlrr(1) = chmlr(1);
// WRF:18813
}
// WRF:18814
} else if((ihmlt==0)) {
// WRF:18815
chmlrr(1) = chmlr(1);
// WRF:18816
}
// WRF:18818
} else {
// WRF:18819
chmlrr(1) = lake_min(chmlrr(1),lake_div((rho0(1)*qhmlr(1)),((xdn(1,lr)*coeff_xvmx{}(lr)))));
// WRF:18820
}
// WRF:18822
}
// WRF:18824
if(((((lhl>1)&&(lhlw<1))&&(!mixedphase))&&(qhlmlr(1)<0.0f))) {
// WRF:18826
if(((ibinhlmlr==0)||(lzhl<1))) {
// WRF:18831
chlmlr(1) = ((lake_div(cx(1,lhl),((qx(1,lhl)+1.e-20f))))*qhlmlr(1));
// WRF:18832
if(((imltshddmr==3)&&(qhlmlr(1)<(-coeff_qxmin{}(lhl))))) {
// WRF:18842
tmp = (1.f+alpha(1,lhl));
// WRF:18843
i = ((int)((dgami*(tmp))));
// WRF:18844
del = (tmp-(dgam*i));
// WRF:18845
g1palp = (gmoi(i)+((((gmoi((i+1))-gmoi(i)))*del)*dgami));
// WRF:18847
ratio = lake_min(maxratiolu,lake_div(mltdiam1,xdia(1,lhl,1)));
// WRF:18849
x = lake_div(gamxinfdp((2.f+alpha(1,lhl)),ratio),g1palp);
// WRF:18850
y = lake_div(gamxinfdp(((2.5f+alpha(1,lhl))+(0.5f*bxx(1,lhl))),ratio),g1palp);
// WRF:18852
hwvent1 = ((0.78f*x)+(y*hlventy(1)));
// WRF:18854
qhlmlr1 = lake_min((((fmlt1(1)*cx(1,lhl))*hwvent1)*xdia(1,lhl,1)),0.0f);
// WRF:18856
chlmlr(1) = ((lake_div(cx(1,lhl),((qx(1,lhl)+1.e-20f))))*lake_min(0.0f,(qhlmlr(1)-qhlmlr1)));
// WRF:18858
}
// WRF:18860
}
// WRF:18862
if(((ibinhlmlr==0)||(lzhl<1))) {
// WRF:18863
if((ihmlt==1)) {
// WRF:18864
chlmlrr(1) = lake_min(chlmlr(1),lake_div((rho0(1)*qhlmlr(1)),((xdn(1,lr)*vmlt))));
// WRF:18865
} else if((ihmlt==2)) {
// WRF:18866
if(((xv(1,lhl)>0.0f)&&(chlmlr(1)<0.0f))) {
// WRF:18869
if((imltshddmr==1)) {
// WRF:18870
tmp = lake_div(((-rho0(1))*qhlmlr(1)),(lake_min((xdn(1,lr)*coeff_xvmx{}(lr)),(xdn(1,lhl)*xv(1,lhl)))));
// WRF:18871
tmp2 = lake_div(((-rho0(1))*qhlmlr(1)),((xdn(1,lr)*vr3mm)));
// WRF:18872
chlmlrr(1) = (lake_div((tmp*((20.e-3f-xdia(1,lhl,3)))),((20.e-3f-sheddiam)))+lake_div((tmp2*((xdia(1,lhl,3)-sheddiam))),((20.e-3f-sheddiam))));
// WRF:18873
chlmlrr(1) = (-lake_max(tmp,lake_min(tmp2,chlmlrr(1))));
// WRF:18874
} else if(((imltshddmr==2)||(imltshddmr==3))) {
// WRF:18877
chlmlrr(1) = lake_div((rho0(1)*qhlmlr(1)),((xdn(1,lr)*vshdgs(1,lhl))));
// WRF:18878
} else {
// WRF:18879
chlmlrr(1) = lake_div((rho0(1)*qhlmlr(1)),(lake_min((xdn(1,lr)*coeff_xvmx{}(lr)),(xdn(1,lhl)*xv(1,lhl)))));
// WRF:18880
}
// WRF:18881
} else {
// WRF:18882
chlmlrr(1) = chlmlr(1);
// WRF:18883
}
// WRF:18884
} else if((ihmlt==0)) {
// WRF:18885
chlmlrr(1) = chlmlr(1);
// WRF:18886
}
// WRF:18888
} else {
// WRF:18889
chlmlrr(1) = lake_min(chlmlrr(1),lake_div((rho0(1)*qhlmlr(1)),((xdn(1,lr)*coeff_xvmx{}(lr)))));
// WRF:18890
}
// WRF:18893
if((((ipconc>=8)&&(lzhl>1))&&(ibinhlmlr<=0))) {
// WRF:18894
if((cx(1,lhl)>0.0f)) {
// WRF:18896
tmp = lake_div(qx(1,lhl),cx(1,lhl));
// WRF:18897
alp = alpha(1,lhl);
// WRF:18899
g1 = g1x(1,lhl);
// WRF:18901
zhlmlr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*((tmp*qhlmlr(1))));
// WRF:18902
}
// WRF:18903
}
// WRF:18904
}
// WRF:18906
}
// WRF:18911
}
// WRF:18912
}
// WRF:18917
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18919
rwcap(1) = ((0.5f)*xdia(1,lr,1));
// WRF:18920
swcap(1) = ((0.5f)*xdia(1,ls,1));
// WRF:18921
hwcap(1) = ((0.5f)*xdia(1,lh,1));
// WRF:18922
if((lhl>1)) {hlcap(1) = ((0.5f)*xdia(1,lhl,1));}
// WRF:18924
if(((qx(1,li)>coeff_qxmin{}(li))&&(xdia(1,li,1)>0.0f))) {
// WRF:18928
cilen(1) = (0.4764f*lake_pow((xdia(1,li,1)),(0.958f)));
// WRF:18929
cval = xdia(1,li,1);
// WRF:18930
aval = cilen(1);
// WRF:18931
eval = sqrtf((1.0f-lake_div((lake_pow(aval,2)),(lake_pow(cval,2)))));
// WRF:18932
fval = lake_min(0.99f,eval);
// WRF:18933
gval = gfk_log(lake_abs(lake_div(((1.f+fval)),((1.f-fval)))));
// WRF:18934
cicap(1) = lake_div((cval*fval),gval);
// WRF:18935
} else {
// WRF:18936
cicap(1) = 0.0f;
// WRF:18937
}
// WRF:18938
}
// WRF:18941
for(int _s0=1;_s0<=ngs;++_s0) {qhdsv(_s0) = 0.0f;}
// WRF:18942
for(int _s0=1;_s0<=ngs;++_s0) {qhldsv(_s0) = 0.0f;}
// WRF:18944
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18945
if((((icond==1)||(temg(1)<=tfrh))||(((qx(1,lr)<=coeff_qxmin{}(lr))&&(qx(1,lc)<=coeff_qxmin{}(lc)))))) {
// WRF:18947
qidsv(1) = ((((fvds(1)*cx(1,li))*civent(1))*cicap(1))*depfac);
// WRF:18949
qsdsv(1) = ((((fvds(1)*cx(1,ls))*swvent(1))*swcap(1))*depfac);
// WRF:18957
} else {
// WRF:18958
qidsv(1) = 0.0f;
// WRF:18959
qsdsv(1) = 0.0f;
// WRF:18960
}
// WRF:18961
qhdsv(1) = ((((fvds(1)*cx(1,lh))*hwvent(1))*hwcap(1))*depfac);
// WRF:18964
if((lhl>1)) {qhldsv(1) = ((((fvds(1)*cx(1,lhl))*hlvent(1))*hlcap(1))*depfac);}
// WRF:18967
}
// WRF:18978
if(dosublimationfix) {
// WRF:18980
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:18982
qitmp(1) = ((qx(1,li)+qx(1,ls))+qx(1,lh));
// WRF:18983
if((lis>1)) {qitmp(1) = (qitmp(1)+qx(1,lis));}
// WRF:18984
if((lhl>1)) {qitmp(1) = (qitmp(1)+qx(1,lhl));}
// WRF:18985
qrtmp(1) = qx(1,lr);
// WRF:18986
qctmp(1) = qx(1,lc);
// WRF:18987
qsimxdep(1) = 0.0f;
// WRF:18988
qsimxsub(1) = 0.0f;
// WRF:18989
dqcitmp(1) = 0.0f;
// WRF:18993
if((qitmp(1)>coeff_qxmin{}(li))) {
// WRF:18995
qitmp1 = qitmp(1);
// WRF:18996
qctmp1 = qctmp(1);
// WRF:18997
felvcptmp = felvcp(1);
// WRF:18998
felscptmp = felscp(1);
// WRF:18999
qvtmp(1) = qx(1,lv);
// WRF:19000
qss(1) = qvs(1);
// WRF:19001
qsstmp = qvs(1);
// WRF:19002
qvstmp = qvs(1);
// WRF:19003
qisstmp = qis(1);
// WRF:19004
thetatmp = theta(1);
// WRF:19005
thetaptmp = thetap(1);
// WRF:19006
temgtmp = temg(1);
// WRF:19007
temcgtmp = temcg(1);
// WRF:19008
qvaptmp = qx(1,lv);
// WRF:19009
qvptmp = 0.0f;
// WRF:19011
qsstmp = qisstmp;
// WRF:19014
dqwvtmp(1) = ((qvtmp(1)-qsstmp));
// WRF:19016
for(itertd=1;itertd<=2;itertd+=1) {
// WRF:19021
if((itertd==1)) {
// WRF:19023
} else {
// WRF:19024
dqcitmp(1) = dqci(1);
// WRF:19026
}
// WRF:19028
dqcw(1) = 0.0f;
// WRF:19029
dqci(1) = 0.0f;
// WRF:19030
dqwv(1) = ((qvtmp(1)-qsstmp));
// WRF:19034
if((dqwv(1)<0.f)) {
// WRF:19035
if((qitmp(1)>(-dqwv(1)))) {
// WRF:19036
dqci(1) = dqwv(1);
// WRF:19037
dqwv(1) = 0.f;
// WRF:19038
} else {
// WRF:19039
dqci(1) = (-qitmp(1));
// WRF:19040
dqwv(1) = (dqwv(1)+qitmp(1));
// WRF:19041
}
// WRF:19043
qvptmp = (qvptmp-((dqcw(1)+dqci(1))));
// WRF:19045
if(((itertd==2)&&(eqtset>1))) {
// WRF:19047
tmp = qitmp(1);
// WRF:19049
cvm = (((cv+(cvv*qvtmp(1)))+(cpl*((qx(1,lc)+qrtmp(1)))))+(cpigb*(tmp)));
// WRF:19052
felvcptmp = lake_div(((felv(1)-(rw*temg(1)))),cvm);
// WRF:19053
felscptmp = lake_div(((fels(1)-(rw*temg(1)))),cvm);
// WRF:19054
}
// WRF:19058
qctmp(1) = (qctmp(1)+dqcw(1));
// WRF:19059
qitmp(1) = (qitmp(1)+dqci(1));
// WRF:19060
thetaptmp = (thetaptmp+(lake_div(1.f,pi0(1))*(((felvcp(1)*dqcw(1))+(felscp(1)*dqci(1))))));
// WRF:19065
}
// WRF:19069
if((dqwv(1)>=0.f)) {
// WRF:19074
fracl(1) = 0.0f;
// WRF:19075
fraci(1) = 1.0f;
// WRF:19076
if(((temg(1)<tfr)&&(temg(1)>thnuc))) {
// WRF:19079
}
// WRF:19080
if((temg(1)<=thnuc)) {
// WRF:19081
fraci(1) = 1.0f;
// WRF:19082
fracl(1) = 0.0f;
// WRF:19083
}
// WRF:19086
gamss = lake_div((((felvcp(1)*fracl(1))+(felscp(1)*fraci(1)))),(pi0(1)));
// WRF:19089
dqvcnd(1) = lake_div(dqwv(1),((1.f+lake_div((fcqv2(1)*qsstmp),(lake_pow(((temg(1)-cbi)),2))))));
// WRF:19092
if((temg(1)>=tfr)) {
// WRF:19093
dqvcnd(1) = lake_div(dqwv(1),((1.f+lake_div((fcqv1(1)*qsstmp),(lake_pow(((temg(1)-cbw)),2))))));
// WRF:19095
}
// WRF:19097
delqci1 = qx(1,li);
// WRF:19100
dqcw(1) = (dqvcnd(1)*fracl(1));
// WRF:19101
dqci(1) = (dqvcnd(1)*fraci(1));
// WRF:19103
thetaptmp = (thetaptmp+lake_div((((felvcp(1)*dqcw(1))+(felscp(1)*dqci(1)))),(pi0(1))));
// WRF:19107
qvptmp = (qvptmp-(dqvcnd(1)));
// WRF:19108
qctmp(1) = (qctmp(1)+dqcw(1));
// WRF:19109
qitmp(1) = (qitmp(1)+dqci(1));
// WRF:19111
if(((itertd==2)&&(eqtset>1))) {
// WRF:19113
tmp = qitmp(1);
// WRF:19115
cvm = (((cv+(cvv*qvtmp(1)))+(cpl*((qctmp(1)+qrtmp(1)))))+(cpigb*(tmp)));
// WRF:19118
felvcptmp = lake_div(((felv(1)-(rw*temg(1)))),cvm);
// WRF:19119
felscptmp = lake_div(((fels(1)-(rw*temg(1)))),cvm);
// WRF:19120
}
// WRF:19122
if((eqtset>2)) {
// WRF:19123
pipert(1) = (pipert(1)+((((0+(felspi(1)*dqci(1)))+(felvpi(1)*dqcw(1))))*dtp));
// WRF:19126
}
// WRF:19130
}
// WRF:19134
if((itertd==1)) {
// WRF:19137
thetatmp = (thetaptmp+theta0(1));
// WRF:19138
temgtmp = (thetatmp*pk(1));
// WRF:19139
qvaptmp = lake_max(((qvptmp+qv0(1))),0.0f);
// WRF:19140
temcgtmp = (temgtmp-tfr);
// WRF:19141
tqvcon = (temgtmp-cbw);
// WRF:19142
ltemq = (lake_div(((temgtmp-163.15f)),fqsat)+1.5f);
// WRF:19143
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:19144
qvstmp = (pqs(1)*tabqvs(ltemq));
// WRF:19145
qisstmp = (pqs(1)*tabqis(ltemq));
// WRF:19146
qctmp(1) = lake_max(0.0f,qctmp(1));
// WRF:19147
qitmp(1) = lake_max(0.0f,qitmp(1));
// WRF:19148
qvtmp(1) = lake_max(0.0f,qvaptmp);
// WRF:19151
qsstmp = qisstmp;
// WRF:19153
} else {
// WRF:19155
qctmp(1) = lake_max(0.0f,qctmp(1));
// WRF:19156
qitmp(1) = lake_max(0.0f,qitmp(1));
// WRF:19158
if((qitmp(1)<qitmp1)) {
// WRF:19159
qsimxsub(1) = (((qitmp1-qitmp(1)))*dtpinv);
// WRF:19160
} else if((qitmp(1)>qitmp1)) {
// WRF:19161
qsimxdep(1) = (((qitmp(1)-qitmp1))*dtpinv);
// WRF:19162
}
// WRF:19165
}
// WRF:19171
}
// WRF:19173
}
// WRF:19175
}
// WRF:19177
} else {
// WRF:19179
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19180
qsimxdep(1) = qvimxd(1);
// WRF:19181
qsimxsub(1) = 1.e20f;
// WRF:19182
}
// WRF:19184
}
// WRF:19188
for(int _s0=1;_s0<=ngs;++_s0) {qhcev(_s0) = 0.0f;}
// WRF:19189
for(int _s0=1;_s0<=ngs;++_s0) {chcev(_s0) = 0.0f;}
// WRF:19190
for(int _s0=1;_s0<=ngs;++_s0) {qhlcev(_s0) = 0.0f;}
// WRF:19191
for(int _s0=1;_s0<=ngs;++_s0) {chlcev(_s0) = 0.0f;}
// WRF:19192
for(int _s0=1;_s0<=ngs;++_s0) {qfcev(_s0) = 0.0f;}
// WRF:19194
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19195
qisbv(1) = 0.0f;
// WRF:19196
qssbv(1) = 0.0f;
// WRF:19197
qidpv(1) = 0.0f;
// WRF:19198
qsdpv(1) = 0.0f;
// WRF:19199
qhsbv(1) = 0.0f;
// WRF:19200
qscev(1) = 0.0f;
// WRF:19201
cscev(1) = 0.0f;
// WRF:19202
if((((icond==1)||(temg(1)<=tfrh))||(((qx(1,lr)<=coeff_qxmin{}(lr))&&(qx(1,lc)<=coeff_qxmin{}(lc)))))) {
// WRF:19207
qisbv(1) = lake_max(lake_min(qidsv(1),0.0f),lake_min((-qimxd(1)),(((-0.5f)*qx(1,li))*dtpinv)));
// WRF:19208
if(((temg(1)<tfr)||(!(qsmlr(1)<0.0f)))) {
// WRF:19209
qssbv(1) = lake_max(lake_min(qsdsv(1),0.0f),lake_min((-qsmxd(1)),(((-0.5f)*qx(1,ls))*dtpinv)));
// WRF:19210
}
// WRF:19211
qidpv(1) = lake_max(qidsv(1),0.0f);
// WRF:19212
qsdpv(1) = lake_max(qsdsv(1),0.0f);
// WRF:19214
if(((qsmlr(1)<0.0f)&&(!mixedphase))) {
// WRF:19216
qscev(1) = lake_div(((((((evapfac*4.0f)*pi)*((qx(1,lv)-qss0(1))))*cx(1,ls))*swcap(1))*swvent(1)),((qss0(1)*((fav(1)+fbv(1))))));
// WRF:19218
qscev(1) = lake_max(lake_min(0.0f,qscev(1)),lake_min((-qsmxd(1)),(((-0.5f)*qx(1,ls))*dtpinv)));
// WRF:19219
} else {
// WRF:19221
}
// WRF:19225
} else {
// WRF:19226
qisbv(1) = 0.0f;
// WRF:19227
qssbv(1) = 0.0f;
// WRF:19228
qidpv(1) = 0.0f;
// WRF:19229
qsdpv(1) = 0.0f;
// WRF:19230
}
// WRF:19232
qhsbv(1) = 0.0f;
// WRF:19233
qhdpv(1) = 0.0f;
// WRF:19234
if((qx(1,lh)>coeff_qxmin{}(lh))) {
// WRF:19235
if(((temg(1)<tfr)||(!(qhmlr(1)<0.0f)))) {
// WRF:19237
qhsbv(1) = lake_max(lake_min(qhdsv(1),0.0f),(-qhmxd(1)));
// WRF:19238
qhdpv(1) = lake_max(qhdsv(1),0.0f);
// WRF:19239
}
// WRF:19241
if(((true&&(qhmlr(1)<0.0f))&&(!mixedphase))) {
// WRF:19247
qhcev(1) = lake_div(((((((evapfac*2.0f)*pi)*((qx(1,lv)-qss0(1))))*cx(1,lh))*xdia(1,lh,1))*hwvent(1)),((qss0(1)*((fav(1)+fbv(1))))));
// WRF:19250
qhcev(1) = lake_max(qhcev(1),(-qhmxd(1)));
// WRF:19251
if((temg(1)>tfr)) {qhcev(1) = lake_min(0.0f,qhcev(1));}
// WRF:19253
}
// WRF:19254
}
// WRF:19257
qhlsbv(1) = 0.0f;
// WRF:19258
qhldpv(1) = 0.0f;
// WRF:19259
if((lhl>1)) {
// WRF:19260
if((qx(1,lhl)>coeff_qxmin{}(lhl))) {
// WRF:19261
if(((temg(1)<tfr)||(!(qhlmlr(1)<0.0f)))) {
// WRF:19262
qhlsbv(1) = lake_max(lake_min(qhldsv(1),0.0f),(-qxmxd(1,lhl)));
// WRF:19263
qhldpv(1) = lake_max(qhldsv(1),0.0f);
// WRF:19264
}
// WRF:19265
if(((qhlmlr(1)<0.0f)&&(!mixedphase))) {
// WRF:19267
qhlcev(1) = lake_div(((((((evapfac*2.0f)*pi)*((qx(1,lv)-qss0(1))))*cx(1,lhl))*xdia(1,lhl,1))*hlvent(1)),((qss0(1)*((fav(1)+fbv(1))))));
// WRF:19270
qhlcev(1) = lake_max(qhlcev(1),(-qhlmxd(1)));
// WRF:19271
if((temg(1)>tfr)) {qhlcev(1) = lake_min(0.0f,qhlcev(1));}
// WRF:19273
}
// WRF:19274
}
// WRF:19275
}
// WRF:19277
temp1 = (((qidpv(1)+qsdpv(1))+qhdpv(1))+qhldpv(1));
// WRF:19283
if((temp1>qsimxdep(1))) {
// WRF:19284
frac = lake_div(qsimxdep(1),temp1);
// WRF:19286
qidpv(1) = (frac*qidpv(1));
// WRF:19287
qsdpv(1) = (frac*qsdpv(1));
// WRF:19288
qhdpv(1) = (frac*qhdpv(1));
// WRF:19289
qhldpv(1) = (frac*qhldpv(1));
// WRF:19296
}
// WRF:19298
temp1 = (((qisbv(1)+qssbv(1))+qhsbv(1))+qhlsbv(1));
// WRF:19301
if((temp1<(-qsimxsub(1)))) {
// WRF:19302
frac = lake_div((-qsimxsub(1)),temp1);
// WRF:19304
qisbv(1) = (frac*qisbv(1));
// WRF:19305
qssbv(1) = (frac*qssbv(1));
// WRF:19306
qhsbv(1) = (frac*qhsbv(1));
// WRF:19307
qhlsbv(1) = (frac*qhlsbv(1));
// WRF:19314
}
// WRF:19317
}
// WRF:19320
if((ipconc>=1)) {
// WRF:19321
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19322
cssbv(1) = ((lake_div(cx(1,ls),((qx(1,ls)+1.e-20f))))*qssbv(1));
// WRF:19323
cisbv(1) = ((lake_div(cx(1,li),((qx(1,li)+1.e-20f))))*qisbv(1));
// WRF:19324
chsbv(1) = ((lake_div(cx(1,lh),((qx(1,lh)+1.e-20f))))*qhsbv(1));
// WRF:19325
if((lhl>1)) {chlsbv(1) = ((lake_div(cx(1,lhl),((qx(1,lhl)+1.e-20f))))*qhlsbv(1));}
// WRF:19326
csdpv(1) = 0.0f;
// WRF:19327
cidpv(1) = 0.0f;
// WRF:19328
cisdpv(1) = 0.0f;
// WRF:19329
chdpv(1) = 0.0f;
// WRF:19330
chldpv(1) = 0.0f;
// WRF:19331
}
// WRF:19332
}
// WRF:19337
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:19338
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19339
qscni(1) = 0.0f;
// WRF:19340
cscni(1) = 0.0f;
// WRF:19341
cscnis(1) = 0.0f;
// WRF:19342
if((((ipconc>=4)&&(iscni>=1))&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:19343
if((iscni==1)) {
// WRF:19344
qscni(1) = lake_div(((((((pi*rho0(1))*(lake_div((0.25f),(6.0f))))*eii(1))*(lake_pow(qx(1,li),2)))*(xdia(1,li,2)))*vtxbar(1,li,1)),xmas(1,li));
// WRF:19348
cscni(1) = lake_min(cimxd(1),lake_div((qscni(1)*rho0(1)),xmas(1,li)));
// WRF:19349
cscnis(1) = (0.5f*cscni(1));
// WRF:19350
} else if((((iscni==2)||(iscni==4))||(iscni==5))) {
// WRF:19351
if((((iscni!=5)&&(qidpv(1)>0.0f))&&(xdia(1,li,3)>=100.e-6f))) {
// WRF:19356
qscni(1) = (lake_min(0.5f,lake_div(xdia(1,li,3),200.e-6f))*qidpv(1));
// WRF:19360
cscni(1) = lake_div(((fscni*qscni(1))*rho0(1)),lake_max((rho_qs*coeff_xvmn{}(ls)),xmas(1,li)));
// WRF:19364
cscnis(1) = cscni(1);
// WRF:19368
}
// WRF:19370
if((iscni!=4)) {
// WRF:19373
tmp = (((((ess(1)*rvt)*aa2)*cx(1,li))*cx(1,li))*xv(1,li));
// WRF:19378
qscni(1) = (qscni(1)+lake_min(qxmxd(1,li),(((2.0f*tmp)*xmas(1,li))*rhoinv(1))));
// WRF:19379
cscni(1) = (cscni(1)+lake_min(cxmxd(1,li),(2.0f*tmp)));
// WRF:19380
cscnis(1) = (cscnis(1)+lake_min(cxmxd(1,li),tmp));
// WRF:19381
}
// WRF:19382
} else if((iscni==3)) {
// WRF:19383
qscni(1) = ((0.001f*eii(1))*lake_max(((qx(1,li)-1.e-3f)),0.0f));
// WRF:19384
qscni(1) = lake_min(qscni(1),qxmxd(1,li));
// WRF:19385
cscni(1) = lake_div((qscni(1)*rho0(1)),xmas(1,li));
// WRF:19386
cscnis(1) = (0.5f*cscni(1));
// WRF:19388
}
// WRF:19390
} else if((ipconc<4)) {
// WRF:19391
if(lwsm6) {
// WRF:19392
qimax = (rhoinv(1)*roqimax);
// WRF:19393
qscni(1) = lake_min((0.90f*qx(1,li)),lake_max(0.0f,(((qx(1,li)-qimax))*dtpinv)));
// WRF:19394
} else {
// WRF:19395
qscni(1) = ((0.001f*eii(1))*lake_max(((qx(1,li)-1.e-3f)),0.0f));
// WRF:19396
qscni(1) = lake_min(qscni(1),qxmxd(1,li));
// WRF:19397
}
// WRF:19398
} else {
// WRF:19399
if(((iscni>0)&&(qx(1,li)>coeff_qxmin{}(li)))) {
// WRF:19400
qscni(1) = lake_div(((((((pi*rho0(1))*(lake_div((0.25f),(6.0f))))*eii(1))*(lake_pow(qx(1,li),2)))*(xdia(1,li,2)))*vtxbar(1,li,1)),xmas(1,li));
// WRF:19404
cscni(1) = lake_min(cimxd(1),lake_div((qscni(1)*rho0(1)),xmas(1,li)));
// WRF:19405
}
// WRF:19407
}
// WRF:19408
}
// WRF:19418
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19420
qsdry(1) = ((qsacr(1)+qsacw(1))+qsaci(1));
// WRF:19423
qhdry(1) = (((qhaci(1)+qhacs(1))+qhacr(1))+qhacw(1));
// WRF:19428
qhldry(1) = 0.0f;
// WRF:19429
if((lhl>1)) {
// WRF:19430
qhldry(1) = (((qhlaci(1)+qhlacs(1))+qhlacr(1))+qhlacw(1));
// WRF:19433
}
// WRF:19434
}
// WRF:19438
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19440
if(((tfrdry<temg(1))&&(temg(1)<tfr))) {
// WRF:19451
if((incwet==0)) {
// WRF:19452
qhwet(1) = (((((xdia(1,lh,1)*hwvent(1))*cx(1,lh))*fwet1(1))+(fwet2(1)*((qhaci(1)+qhacs(1))))));
// WRF:19455
qhwet(1) = lake_max(0.0f,qhwet(1));
// WRF:19456
} else {
// WRF:19457
}
// WRF:19462
qhlwet(1) = 0.0f;
// WRF:19463
if((lhl>1)) {
// WRF:19464
if((incwet==0)) {
// WRF:19465
qhlwet(1) = (((((xdia(1,lhl,1)*hlvent(1))*cx(1,lhl))*fwet1(1))+(fwet2(1)*((qhlaci(1)+qhlacs(1))))));
// WRF:19468
qhlwet(1) = lake_max(0.0f,qhlwet(1));
// WRF:19470
} else {
// WRF:19471
}
// WRF:19472
}
// WRF:19474
} else {
// WRF:19476
qhwet(1) = qhdry(1);
// WRF:19477
qhlwet(1) = qhldry(1);
// WRF:19478
}
// WRF:19482
}
// WRF:19487
for(int _s0=1;_s0<=ngs;++_s0) {qsshr(_s0) = 0.0f;}
// WRF:19488
for(int _s0=1;_s0<=ngs;++_s0) {qhshr(_s0) = 0.0f;}
// WRF:19489
for(int _s0=1;_s0<=ngs;++_s0) {qhlshr(_s0) = 0.0f;}
// WRF:19490
for(int _s0=1;_s0<=ngs;++_s0) {qhshh(_s0) = 0.0f;}
// WRF:19491
for(int _s0=1;_s0<=ngs;++_s0) {csshr(_s0) = 0.0f;}
// WRF:19492
for(int _s0=1;_s0<=ngs;++_s0) {csshrr(_s0) = 0.0f;}
// WRF:19493
for(int _s0=1;_s0<=ngs;++_s0) {chshr(_s0) = 0.0f;}
// WRF:19494
for(int _s0=1;_s0<=ngs;++_s0) {chlshr(_s0) = 0.0f;}
// WRF:19495
for(int _s0=1;_s0<=ngs;++_s0) {chshrr(_s0) = 0.0f;}
// WRF:19496
for(int _s0=1;_s0<=ngs;++_s0) {chlshrr(_s0) = 0.0f;}
// WRF:19497
for(int _s0=1;_s0<=ngs;++_s0) {vhshdr(_s0) = 0.0f;}
// WRF:19498
for(int _s0=1;_s0<=ngs;++_s0) {vhlshdr(_s0) = 0.0f;}
// WRF:19499
for(int _s0=1;_s0<=ngs;++_s0) {wetsfc(_s0) = false;}
// WRF:19500
for(int _s0=1;_s0<=ngs;++_s0) {wetgrowth(_s0) = false;}
// WRF:19501
for(int _s0=1;_s0<=ngs;++_s0) {wetsfchl(_s0) = false;}
// WRF:19502
for(int _s0=1;_s0<=ngs;++_s0) {wetgrowthhl(_s0) = false;}
// WRF:19504
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19508
qhshr(1) = lake_min(0.0f,(qhwet(1)-qhdry(1)));
// WRF:19512
qhlshr(1) = lake_min(0.0f,(qhlwet(1)-qhldry(1)));
// WRF:19517
qsshr(1) = 0.0f;
// WRF:19522
if((temg(1)<243.15f)) {
// WRF:19523
qsshr(1) = 0.0f;
// WRF:19524
qhshr(1) = 0.0f;
// WRF:19525
qhlshr(1) = 0.0f;
// WRF:19526
vhshdr(1) = 0.0f;
// WRF:19527
vhlshdr(1) = 0.0f;
// WRF:19528
wetsfc(1) = false;
// WRF:19529
wetgrowth(1) = false;
// WRF:19530
wetsfchl(1) = false;
// WRF:19531
wetgrowthhl(1) = false;
// WRF:19532
}
// WRF:19536
if((temg(1)>tfr)) {
// WRF:19538
if(false) {
// WRF:19539
qsshr(1) = (-qsdry(1));
// WRF:19540
qhshr(1) = (-qhdry(1));
// WRF:19541
qhlshr(1) = (-qhldry(1));
// WRF:19542
} else {
// WRF:19544
qsshr(1) = ((-qsacr(1))-qsacw(1));
// WRF:19545
qhlshr(1) = ((-qhlacw(1))-qhlacr(1));
// WRF:19546
qhshr(1) = ((-qhacw(1))-qhacr(1));
// WRF:19548
}
// WRF:19550
vhshdr(1) = ((-vhacw(1))-vhacr(1));
// WRF:19551
vhlshdr(1) = ((-vhlacw(1))-vhlacr(1));
// WRF:19552
qhwet(1) = 0.0f;
// WRF:19553
qhlwet(1) = 0.0f;
// WRF:19554
}
// WRF:19557
wetsfc(1) = ((((qhshr(1)<0.0f)&&(temg(1)<tfr)))||(((qhmlr(1)<(-coeff_qxmin{}(lh)))&&(temg(1)>tfr))));
// WRF:19558
wetgrowth(1) = (((qhshr(1)<0.0f)&&(temg(1)<tfr)));
// WRF:19560
if(((qhlshr(1)<0.0f)&&(temg(1)<tfr))) {
// WRF:19561
wetsfchl(1) = ((((qhlshr(1)<0.0f)&&(temg(1)<tfr)))||(((qhlmlr(1)<(-coeff_qxmin{}(lhl)))&&(temg(1)>tfr))));
// WRF:19562
wetgrowthhl(1) = (((qhlshr(1)<0.0f)&&(temg(1)<tfr)));
// WRF:19563
}
// WRF:19565
}
// WRF:19567
if((ipconc>=1)) {
// WRF:19568
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19569
csshr(1) = 0.0f;
// WRF:19571
chshr(1) = 0.0f;
// WRF:19577
chshrr(1) = lake_div((rho0(1)*qhshr(1)),((xdn(1,lr)*vshdgs(1,lh))));
// WRF:19581
chlshr(1) = 0.0f;
// WRF:19582
chlshrr(1) = 0.0f;
// WRF:19583
if((lhl>1)) {
// WRF:19587
chlshr(1) = 0.0f;
// WRF:19593
chlshrr(1) = lake_div((rho0(1)*qhlshr(1)),((xdn(1,lr)*vshdgs(1,lhl))));
// WRF:19595
}
// WRF:19598
}
// WRF:19599
}
// WRF:19606
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19610
if((qsshr(1)<0.0f)) {
// WRF:19611
qsdpv(1) = 0.0f;
// WRF:19612
qssbv(1) = 0.0f;
// WRF:19613
} else {
// WRF:19614
qsshr(1) = 0.0f;
// WRF:19615
}
// WRF:19627
if((wetgrowth(1)||(((mixedphase&&(fhw(1)>0.05f))&&(temg(1)>243.15f))))) {
// WRF:19632
if(((coeff_lvol{}(lh)>1)&&(!mixedphase))) {
// WRF:19634
if(iwetsoak) {
// WRF:19636
rimdn(1,lh) = coeff_xdnmx{}(lh);
// WRF:19637
raindn(1,lh) = coeff_xdnmx{}(lh);
// WRF:19638
vhacw(1) = lake_div((qhacw(1)*rho0(1)),rimdn(1,lh));
// WRF:19639
vhacr(1) = lake_div((qhacr(1)*rho0(1)),raindn(1,lh));
// WRF:19641
if((xdn(1,lh)<coeff_xdnmx{}(lh))) {
// WRF:19644
v1 = lake_div((((1.f-lake_div(xdn(1,lh),coeff_xdnmx{}(lh))))*vx(1,lh)),(dtp));
// WRF:19646
v2 = lake_div((rho0(1)*qhwet(1)),coeff_xdnmx{}(lh));
// WRF:19648
vhsoak(1) = lake_min(v1,v2);
// WRF:19651
}
// WRF:19653
}
// WRF:19655
vhshdr(1) = lake_min(0.0f,((lake_div((rho0(1)*qhwet(1)),coeff_xdnmx{}(lh))-vhacw(1))-vhacr(1)));
// WRF:19657
} else if(((coeff_lvol{}(lh)>1)&&mixedphase)) {
// WRF:19660
}
// WRF:19663
qhdpv(1) = 0.0f;
// WRF:19665
chdpv(1) = 0.0f;
// WRF:19670
if((ehi(1)>0.0f)) {
// WRF:19671
qhaci(1) = lake_min(qimxd(1),qhaci0(1));
// WRF:19672
chaci(1) = lake_min(cimxd(1),chaci0(1));
// WRF:19673
}
// WRF:19674
if((ehs(1)>0.0f)) {
// WRF:19676
qhacs(1) = lake_min(qsmxd(1),qhacs0(1));
// WRF:19677
chacs(1) = lake_min(csmxd(1),chacs0(1));
// WRF:19678
ehs(1) = ehsmax;
// WRF:19679
qhacs(1) = lake_min(qsmxd(1),qhacs(1));
// WRF:19680
}
// WRF:19683
wetsfc(1) = true;
// WRF:19685
} else {
// WRF:19687
}
// WRF:19693
if(((lhl>1)&&((wetgrowthhl(1)||(((mixedphase&&(fhlw(1)>0.05f))&&(temg(1)>243.15f))))))) {
// WRF:19697
qhldpv(1) = 0.0f;
// WRF:19699
chldpv(1) = 0.0f;
// WRF:19705
if(((coeff_lvol{}(lhl)>1)&&(!mixedphase))) {
// WRF:19708
if(iwetsoak) {
// WRF:19710
rimdn(1,lhl) = coeff_xdnmx{}(lhl);
// WRF:19711
raindn(1,lhl) = coeff_xdnmx{}(lhl);
// WRF:19712
vhlacw(1) = lake_div((qhlacw(1)*rho0(1)),rimdn(1,lhl));
// WRF:19713
vhlacr(1) = lake_div((qhlacr(1)*rho0(1)),raindn(1,lhl));
// WRF:19715
if((xdn(1,lhl)<coeff_xdnmx{}(lhl))) {
// WRF:19718
v1 = lake_div((((1.f-lake_div(xdn(1,lhl),coeff_xdnmx{}(lhl))))*vx(1,lhl)),(dtp));
// WRF:19720
v2 = lake_div((rho0(1)*qhlwet(1)),coeff_xdnmx{}(lhl));
// WRF:19721
if((v1>v2)) {
// WRF:19722
vhlsoak(1) = v2;
// WRF:19723
} else {
// WRF:19724
vhlsoak(1) = v1;
// WRF:19725
}
// WRF:19728
} else {
// WRF:19729
vhlsoak(1) = 0.0f;
// WRF:19733
}
// WRF:19735
}
// WRF:19737
vhlshdr(1) = lake_min(0.0f,((lake_div((rho0(1)*qhlwet(1)),coeff_xdnmx{}(lhl))-vhlacw(1))-vhlacr(1)));
// WRF:19740
} else if(((coeff_lvol{}(lhl)>1)&&mixedphase)) {
// WRF:19743
}
// WRF:19745
if((ehli(1)>0.0f)) {
// WRF:19746
qhlaci(1) = lake_min(qimxd(1),qhlaci0(1));
// WRF:19747
chlaci(1) = lake_min(cimxd(1),chlaci0(1));
// WRF:19748
}
// WRF:19753
if((ehls(1)>0.0f)) {
// WRF:19754
qhlacs(1) = lake_min(qsmxd(1),qhlacs0(1));
// WRF:19755
chlacs(1) = lake_min(csmxd(1),chlacs0(1));
// WRF:19756
ehls(1) = ehsmax;
// WRF:19758
}
// WRF:19764
wetsfchl(1) = true;
// WRF:19767
} else {
// WRF:19770
}
// WRF:19772
}
// WRF:19776
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19778
qhcni(1) = 0.0f;
// WRF:19779
chcni(1) = 0.0f;
// WRF:19780
chcnih(1) = 0.0f;
// WRF:19781
vhcni(1) = 0.0f;
// WRF:19783
if((iglcnvi>=1)) {
// WRF:19784
if(((temg(1)<273.0f)&&((qiacw(1)-qidpv(1))>0.0f))) {
// WRF:19787
tmp = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*vtxbar(1,li,1)))),((temg(1)-273.15f)))),(rimc2)));
// WRF:19790
tmp = lake_min(lake_max(rimc3,tmp),900.0f);
// WRF:19798
if(((tmp>=200.0f)||(iglcnvi>=2))) {
// WRF:19799
r = lake_max((0.5f*((xdn(1,li)+tmp))),coeff_xdnmn{}(lh));
// WRF:19801
qhcni(1) = ((qiacw(1)-qidpv(1)));
// WRF:19802
chcni(1) = lake_div((cx(1,li)*qhcni(1)),qx(1,li));
// WRF:19804
chcnih(1) = lake_min(chcni(1),lake_div((rho0(1)*qhcni(1)),((r*coeff_xvmn{}(lh)))));
// WRF:19806
vhcni(1) = lake_div((rho0(1)*qhcni(1)),r);
// WRF:19807
}
// WRF:19809
} else if((iglcnvi==3)) {
// WRF:19811
if((((temg(1)<273.0f)&&((qiacw(1)*dtp)>(2.f*coeff_qxmin{}(lh))))&&((gamice73fac*xmas(1,li))>(coeff_xdnmn{}(lh)*coeff_xvmn{}(lh))))) {
// WRF:19814
tmp = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*vtxbar(1,li,1)))),((temg(1)-273.15f)))),(rimc2)));
// WRF:19817
tmp = lake_min(lake_max(rimc3,tmp),900.0f);
// WRF:19827
if((tmp>=coeff_xdnmn{}(lh))) {
// WRF:19828
r = lake_max((0.5f*((xdn(1,li)+tmp))),coeff_xdnmn{}(lh));
// WRF:19830
qhcni(1) = (0.5f*qiacw(1));
// WRF:19831
chcni(1) = lake_div(qhcni(1),((gamice73fac*xmas(1,li))));
// WRF:19832
chcnih(1) = lake_min(chcni(1),lake_div((rho0(1)*qhcni(1)),((r*coeff_xvmn{}(lh)))));
// WRF:19834
vhcni(1) = lake_div((rho0(1)*qhcni(1)),r);
// WRF:19835
}
// WRF:19837
}
// WRF:19840
}
// WRF:19841
}
// WRF:19844
}
// WRF:19847
for(int _s0=1;_s0<=ngs;++_s0) {qhlcnh(_s0) = 0.0f;}
// WRF:19848
for(int _s0=1;_s0<=ngs;++_s0) {chlcnh(_s0) = 0.0f;}
// WRF:19849
for(int _s0=1;_s0<=ngs;++_s0) {chlcnhhl(_s0) = 0.0f;}
// WRF:19850
for(int _s0=1;_s0<=ngs;++_s0) {vhlcnh(_s0) = 0.0f;}
// WRF:19851
for(int _s0=1;_s0<=ngs;++_s0) {vhlcnhl(_s0) = 0.0f;}
// WRF:19852
for(int _s0=1;_s0<=ngs;++_s0) {zhlcnh(_s0) = 0.0f;}
// WRF:19854
for(int _s0=1;_s0<=ngs;++_s0) {qhcnhl(_s0) = 0.0f;}
// WRF:19855
for(int _s0=1;_s0<=ngs;++_s0) {chcnhl(_s0) = 0.0f;}
// WRF:19856
for(int _s0=1;_s0<=ngs;++_s0) {vhcnhl(_s0) = 0.0f;}
// WRF:19857
for(int _s0=1;_s0<=ngs;++_s0) {zhcnhl(_s0) = 0.0f;}
// WRF:19860
if((lhl>1)) {
// WRF:19862
if(((ihlcnh==1)||(ihlcnh==3))) {
// WRF:19867
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:19872
if((hlcnhdia>0)) {
// WRF:19873
ltest = (xdia(1,lh,3)>hlcnhdia);
// WRF:19874
} else {
// WRF:19876
ltest = ((xdia(1,lh,1)*((4.f+alpha(1,lh))))>lake_abs(hlcnhdia));
// WRF:19877
}
// WRF:19879
if(((iusedw==0)&&(ihlcnh==1))) {
// WRF:19880
dg0(1) = (-1.f);
// WRF:19881
} else {
// WRF:19882
if((((((((((qhacw(1)+qhacr(1)))*dtp)>coeff_qxmin{}(lh))&&(qx(1,lh)>hlcnhqmin))&&(temg(1)<=(tfr-2.0f)))&&(temg(1)>dwtempmin)))||((wetgrowth(1)&&(qx(1,lh)>hlcnhqmin))))) {
// WRF:19887
x = ((((1.1e4f*rho0(1))*(((ehw(1)*qx(1,lc))+(ehr(1)*qx(1,lr)))))-((1.3e3f*rho0(1))*qx(1,li)))+1.0f);
// WRF:19889
if((x>1.e-20f)) {
// WRF:19890
arg = lake_min(70.0f,(lake_div((-temcg(1)),x)));
// WRF:19891
dwr = (0.01f*((gfk_exp(arg)-1.0f)));
// WRF:19892
} else {
// WRF:19893
dwr = 1.e30f;
// WRF:19894
}
// WRF:19895
d = dwr;
// WRF:19896
if((((dwr<0.2f)&&(dwr>0.0f))&&((rho0(1)*((qx(1,lc)+qx(1,lr))))>1.e-4f))) {
// WRF:19897
sqrtrhovt = sqrtf(rhovt(1));
// WRF:19898
fventh = ((sqrtrhovt*(lake_pow(fpndl(1),(lake_div(1.f,3.f)))))*lake_pow((fakvisc(1)),((-0.5f))));
// WRF:19899
fventm = ((sqrtrhovt*(lake_pow(fschm(1),(lake_div(1.f,3.f)))))*lake_pow((fakvisc(1)),((-0.5f))));
// WRF:19900
ltemq = (lake_div(((tfr-163.15f)),fqsat)+1.5f);
// WRF:19901
qvs0 = (pqs(1)*tabqvs(ltemq));
// WRF:19902
denomdp = (felf(1)+(fcw(1)*temcg(1)));
// WRF:19903
denominvdp = lake_div(1.e0,((felf(1)+(fcw(1)*temcg(1)))));
// WRF:19906
h1 = ((((-ftka(1))*temcg(1))-(((felv(1)*fwvdf(1))*rho0(1))*((qx(1,lv)-qvs0)))));
// WRF:19907
h2 = ((((ehi(1)*qx(1,li))*rho0(1))*fci(1))*temcg(1));
// WRF:19908
h3 = (lake_max(dwehwmin,ehw(1))*qx(1,lc));
// WRF:19909
h4 = (ehr(1)*qx(1,lr));
// WRF:19911
for(n=1;n<=10;n+=1) {
// WRF:19912
d = lake_max(d,1.e-4f);
// WRF:19913
dold = d;
// WRF:19914
vth = (axx(1,lh)*lake_pow(d,bxx(1,lh)));
// WRF:19915
x2 = ((fventh*sqrtrhovt)*sqrtf((d*vth)));
// WRF:19916
if((x2>1.4f)) {
// WRF:19917
ah = (0.78f+(0.308f*x2));
// WRF:19918
} else {
// WRF:19919
ah = (1.0f+(0.108f*lake_pow(x2,2)));
// WRF:19920
}
// WRF:19922
if(false) {
// WRF:19923
x1 = ((fventm*sqrtrhovt)*sqrtf((d*vth)));
// WRF:19924
if((x1>1.4f)) {
// WRF:19925
am = (0.78f+(0.308f*x1));
// WRF:19926
} else {
// WRF:19927
am = (1.0f+(0.108f*lake_pow(x1,2)));
// WRF:19928
}
// WRF:19930
d = lake_div(((8.f*denominvdp)*((((((am*felv(1))*fwvdf(1))*rho0(1))*((qvs0-qx(1,lv))))-((ah*ftka(1))*temcg(1))))),((dtp*((((((lake_max(0.001f,(vth-vtxbar(1,lc,1)))*h3)+(lake_max(0.001f,(vth-vtxbar(1,lr,1)))*h4)))*rho0(1))+((lake_max(0.001f,(vth-vtxbar(1,li,1)))*h2)*denominvdp))))));
// WRF:19935
} else {
// WRF:19939
d = lake_div(((8.f*ah)*h1),(((((((lake_max(0.001f,(vth-vtxbar(1,lc,1)))*h3)+(lake_max(0.001f,(vth-vtxbar(1,lr,1)))*h4)))*rho0(1))*denomdp)+(lake_max(0.001f,(vth-vtxbar(1,li,1)))*h2))));
// WRF:19944
}
// WRF:19945
if(((lake_div(lake_abs((dold-d)),dold)<0.05f)||(((n>3)&&(d>dg0thresh))))) {break;}
// WRF:19947
}
// WRF:19948
}
// WRF:19950
dg0(1) = lake_min(dwmax,lake_max(d,dwmin));
// WRF:19951
} else {
// WRF:19952
if((((qx(1,lh)>coeff_qxmin{}(lh))&&(qx(1,lh)>hlcnhqmin))&&(temg(1)<=(tfr-2.0f)))) {
// WRF:19953
dg0(1) = dwmax;
// WRF:19954
} else {
// WRF:19955
dg0(1) = (dg0thresh+0.0001f);
// WRF:19956
}
// WRF:19957
}
// WRF:19959
if(((((ihlcnh==3)&&((((qhacw(1)+qhacr(1)))*dtp)>coeff_qxmin{}(lh)))&&(qx(1,lh)>hlcnhqmin))&&(temg(1)<=(tfr-2.0f)))) {
// WRF:19962
dg0(1) = lake_min(dg0(1),(dg0thresh-0.0001f));
// WRF:19963
}
// WRF:19965
}
// WRF:19967
wtest = (((dg0(1)>0.0f)&&(dg0(1)<dg0thresh)));
// WRF:19969
if((ihlcnh==1)) {
// WRF:19971
if(((((((wetgrowth(1)&&(((xdn(1,lh)>hldnmn)||(lvh<1))))&&(rimdn(1,lh)>800.f))&&ltest)&&(qx(1,lh)>hlcnhqmin)))||wtest)) {
// WRF:19976
if((((qhacw(1)>0.0f)&&(qhacw(1)>qhaci(1)))&&(temg(1)<=(tfr-2.0f)))) {
// WRF:19980
if(wtest) {
// WRF:19981
dh0 = dg0(1);
// WRF:19982
} else {
// WRF:19983
x = ((((1.1e4f*((rho0(1)*qx(1,lc))))-((1.3e3f*rho0(1))*qx(1,li)))+1.0e-3f));
// WRF:19984
if((x>1.e-20f)) {
// WRF:19985
arg = lake_min(70.0f,(lake_div((-temcg(1)),x)));
// WRF:19986
dh0 = (0.01f*((gfk_exp(arg)-1.0f)));
// WRF:19987
} else {
// WRF:19988
dh0 = 1.e30f;
// WRF:19989
}
// WRF:19990
}
// WRF:19995
if((lake_div(xdia(1,lh,3),dh0)>0.1f)) {
// WRF:19997
tmp = (((qhacw(1)+qhacr(1))+qhaci(1))+qhacs(1));
// WRF:19999
qtmp = (lake_min(100.0f,lake_div(xdia(1,lh,3),((2.0f*dh0))))*(tmp));
// WRF:20000
qhlcnh(1) = lake_min(qxmxd(1,lh),qtmp);
// WRF:20002
if((ipconc>=5)) {
// WRF:20004
if((!wtest)) {dh0 = lake_min(dh0,10.e-3f);}
// WRF:20005
if((qx(1,lhl)>0.1e-3f)) {dh0 = lake_max(dh0,xdia(1,lhl,3));}
// WRF:20006
chlcnhhl(1) = lake_min(cxmxd(1,lh),lake_div((rho0(1)*qhlcnh(1)),(lake_div(((pi*xdn(1,lh))*lake_pow(dh0,3)),6.0f))));
// WRF:20008
r = lake_div((rho0(1)*qhlcnh(1)),((xdn(1,lh)*xv(1,lh))));
// WRF:20009
chlcnh(1) = lake_max(chlcnhhl(1),r);
// WRF:20010
}
// WRF:20012
vhlcnh(1) = lake_div((rho0(1)*qhlcnh(1)),xdn(1,lh));
// WRF:20013
vhlcnhl(1) = lake_div((rho0(1)*qhlcnh(1)),lake_max(coeff_xdnmn{}(lhl),xdn(1,lh)));
// WRF:20015
}
// WRF:20017
}
// WRF:20018
}
// WRF:20020
} else if((ihlcnh==3)) {
// WRF:20023
if((wtest&&(((((qhacw(1)*dtp)>coeff_qxmin{}(lh))&&(temg(1)<(tfr-2.f)))&&(qx(1,lh)>hlcnhqmin))))) {
// WRF:20026
if((ipconc==5)) {
// WRF:20029
}
// WRF:20031
ratio = lake_min(maxratiolu,lake_div(dg0(1),xdia(1,lh,1)));
// WRF:20035
tmp2 = gaminterp(ratio,alpha(1,lh),4,1);
// WRF:20036
if((ipconc==5)) {
// WRF:20038
}
// WRF:20039
qxd1 = (qx(1,lh)*(tmp2));
// WRF:20040
qhlcnh(1) = (dtpinv*qxd1);
// WRF:20041
flim = 1.0f;
// WRF:20042
tmp3 = qxmxd(1,lh);
// WRF:20043
if((qxd1>tmp3)) {
// WRF:20046
}
// WRF:20050
if(((((qxd1>coeff_qxmin{}(lhl))&&(ipconc>5)))||(((qxd1>(10.f*coeff_qxmin{}(lhl)))&&(ipconc==5))))) {
// WRF:20053
tmp = gaminterp(ratio,alpha(1,lh),1,1);
// WRF:20054
if((ipconc==5)) {
// WRF:20056
}
// WRF:20057
cxd1 = ((flim*cx(1,lh))*(tmp));
// WRF:20058
chlcnh(1) = (dtpinv*cxd1);
// WRF:20059
chlcnhhl(1) = chlcnh(1);
// WRF:20061
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(dmhlopt>0))) {
// WRF:20062
tmp = lake_div((rho0(1)*qhlcnh(1)),chlcnhhl(1));
// WRF:20063
if((tmp<xmas(1,lhl))) {
// WRF:20065
dh0 = lake_pow((lake_div((((qxd1*lake_pow(tmp,(lake_div(1.f,3.f))))+(qx(1,lhl)*lake_pow(xmas(1,lhl),(lake_div(1.f,3.f)))))),((qxd1+qx(1,lhl))))),3);
// WRF:20066
chlcnhhl(1) = lake_min(chlcnhhl(1),lake_div((rho0(1)*qhlcnh(1)),dh0));
// WRF:20067
} else {
// WRF:20069
}
// WRF:20070
}
// WRF:20074
if((((ipconc>=6)&&(lzh>1))&&(lzhl>1))) {
// WRF:20075
tmp3 = gaminterp(ratio,alpha(1,lh),11,1);
// WRF:20076
zxd1 = ((flim*zx(1,lh))*(tmp3));
// WRF:20077
zhlcnh(1) = (dtpinv*zxd1);
// WRF:20078
} else {
// WRF:20079
zxd1 = 0;
// WRF:20080
}
// WRF:20082
} else {
// WRF:20083
qhlcnh(1) = 0.0f;
// WRF:20084
}
// WRF:20086
vhlcnh(1) = lake_div((rho0(1)*qhlcnh(1)),xdn(1,lh));
// WRF:20087
vhlcnhl(1) = lake_div((rho0(1)*qhlcnh(1)),lake_max(coeff_xdnmn{}(lhl),xdn(1,lh)));
// WRF:20089
}
// WRF:20092
}
// WRF:20094
}
// WRF:20096
} else if((ihlcnh==2)) {
// WRF:20106
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20107
if(((((qhacw(1)*dtp)>coeff_qxmin{}(lh))&&(temg(1)<(tfr-2.f)))&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:20108
ratio = lake_min(maxratiolu,lake_div(hldia1,xdia(1,lh,1)));
// WRF:20111
tmp = gaminterp(ratio,alpha(1,lh),1,1);
// WRF:20112
cxd1 = (cx(1,lh)*(tmp));
// WRF:20113
chlcnh(1) = (dtpinv*cxd1);
// WRF:20114
chlcnhhl(1) = chlcnh(1);
// WRF:20117
tmp2 = gaminterp(ratio,alpha(1,lh),4,1);
// WRF:20118
qxd1 = (qx(1,lh)*(tmp2));
// WRF:20119
qhlcnh(1) = (dtpinv*qxd1);
// WRF:20122
if(((lzh>1)&&(lzhl>1))) {
// WRF:20123
tmp3 = gaminterp(ratio,alpha(1,lh),11,1);
// WRF:20124
zxd1 = (zx(1,lh)*(tmp3));
// WRF:20125
zhlcnh(1) = (dtpinv*zxd1);
// WRF:20126
} else {
// WRF:20127
zxd1 = 0;
// WRF:20128
}
// WRF:20129
vhlcnh(1) = lake_div((rho0(1)*qhlcnh(1)),xdn(1,lh));
// WRF:20130
vhlcnhl(1) = lake_div((rho0(1)*qhlcnh(1)),lake_max(coeff_xdnmn{}(lhl),xdn(1,lh)));
// WRF:20132
}
// WRF:20134
}
// WRF:20136
} else if((ihlcnh==0)) {
// WRF:20138
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20141
if(((wetgrowth(1)&&(temg(1)<(tfr-5.f)))&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:20142
if(((qhacw(1)>1.e-6f)&&(xdn(1,lh)>700.f))) {
// WRF:20143
qhlcnh(1) = (((lake_div((((pi*xdn(1,lh))*cx(1,lh))),(((6.0f*rho0(1))*dtp))))*gfk_exp(lake_div((-hldia1),xdia(1,lh,1))))*(((((lake_pow(hldia1,3))+((3.0f*(lake_pow(hldia1,2)))*xdia(1,lh,1)))+((6.0f*(hldia1))*(lake_pow(xdia(1,lh,1),2))))+(6.0f*(lake_pow(xdia(1,lh,1),3))))));
// WRF:20148
qhlcnh(1) = lake_min(qhlcnh(1),qhmxd(1));
// WRF:20149
if((ipconc>=5)) {
// WRF:20150
chlcnh(1) = lake_min(cxmxd(1,lh),(cx(1,lh)*gfk_exp(lake_div((-hldia1),xdia(1,lh,1)))));
// WRF:20151
chlcnhhl(1) = chlcnh(1);
// WRF:20153
}
// WRF:20154
vhlcnh(1) = lake_div((rho0(1)*qhlcnh(1)),xdn(1,lh));
// WRF:20155
vhlcnhl(1) = lake_div((rho0(1)*qhlcnh(1)),lake_max(coeff_xdnmn{}(lhl),xdn(1,lh)));
// WRF:20156
}
// WRF:20157
}
// WRF:20158
}
// WRF:20162
}
// WRF:20165
if((icvhl2h>=1)) {
// WRF:20166
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20167
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(xdn(1,lhl)<(0.5f*((coeff_xdnmn{}(lhl)+coeff_xdnmx{}(lhl))))))) {
// WRF:20168
tmp = lake_min(0.95f,(1.f-(0.5f*((1.f+tanhf((0.125f*((xdn(1,lhl)-(1.01f*coeff_xdnmn{}(lhl)))))))))));
// WRF:20169
qhcnhl(1) = ((tmp*qx(1,lhl))*dtpinv);
// WRF:20170
chcnhl(1) = lake_div((cx(1,lhl)*qhcnhl(1)),qx(1,lhl));
// WRF:20171
vhcnhl(1) = lake_div((vx(1,lhl)*qhcnhl(1)),qx(1,lhl));
// WRF:20173
}
// WRF:20174
}
// WRF:20176
}
// WRF:20178
}
// WRF:20186
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20188
qhcns(1) = 0.0f;
// WRF:20189
chcns(1) = 0.0f;
// WRF:20190
chcnsh(1) = 0.0f;
// WRF:20191
vhcns(1) = 0.0f;
// WRF:20193
qscnh(1) = 0.0f;
// WRF:20194
cscnh(1) = 0.0f;
// WRF:20195
vscnh(1) = 0.0f;
// WRF:20197
if((ipconc>=5)) {
// WRF:20200
if(((((temg(1)<tfr)&&(qx(1,lh)>coeff_qxmin{}(lh)))&&(qhdpv(1)>(coeff_qxmin{}(lh)*dtpinv)))&&(qhacw(1)<(coeff_qxmin{}(lh)*dtpinv)))) {
// WRF:20202
if((xdn(1,lh)<290.f)) {
// WRF:20206
}
// WRF:20207
}
// WRF:20210
if(((qx(1,ls)>coeff_qxmin{}(ls))&&(qsacw(1)>0.0f))) {
// WRF:20229
if((iglcnvs==1)) {
// WRF:20231
dnnet = ((cscnvis(1)+cscnis(1))-csacs(1));
// WRF:20232
dqnet = ((((qscnvi(1)+qscni(1))+qsacw(1))+qsdpv(1))+qssbv(1));
// WRF:20234
a3 = lake_div(1.f,((rho0(1)*qx(1,ls))));
// WRF:20235
a1 = gfk_exp(((((-xdn(1,ls))*cx(1,ls))*vgra)*a3));
// WRF:20237
a2 = (((1.f-((((cx(1,ls)*vgra)*xdn(1,ls))*a3))))*dnnet);
// WRF:20239
a4 = (lake_div((((lake_pow(cx(1,ls),2)*vgra)*xdn(1,ls))*a3),qx(1,ls))*dqnet);
// WRF:20241
chcns(1) = lake_max(0.0f,(a1*((a2+a4))));
// WRF:20242
chcns(1) = lake_min(chcns(1),cxmxd(1,ls));
// WRF:20243
chcnsh(1) = chcns(1);
// WRF:20245
qhcns(1) = lake_min((((xdn(1,ls)*vgra)*rhoinv(1))*chcns(1)),qxmxd(1,ls));
// WRF:20246
vhcns(1) = lake_div((rho0(1)*qhcns(1)),lake_max(xdn(1,ls),coeff_xdnmn{}(lh)));
// WRF:20249
} else if((iglcnvs>=2)) {
// WRF:20251
if(((temg(1)<273.0f)&&((((qsacw(1)-qsdpv(1))>0.0f)||((((iglcnvs>=3)&&((qsacw(1)*dtp)>(2.f*coeff_qxmin{}(lh))))&&((gamsnow73fac*xmas(1,ls))>(coeff_xdnmn{}(lh)*coeff_xvmn{}(lh))))))))) {
// WRF:20255
tmp = (rimc1*lake_pow((lake_div(((-((((0.5f)*(1.e+06f))*xdia(1,lc,1))))*(((0.60f)*vtxbar(1,ls,1)))),((temg(1)-273.15f)))),(rimc2)));
// WRF:20259
tmp = lake_min(tmp,900.0f);
// WRF:20267
if((iglcnvs==2)) {
// WRF:20268
if((tmp>=200.0f)) {
// WRF:20269
r = lake_max((0.5f*((xdn(1,ls)+tmp))),coeff_xdnmn{}(lh));
// WRF:20271
qhcns(1) = ((qsacw(1)-qsdpv(1)));
// WRF:20272
chcns(1) = lake_div((cx(1,ls)*qhcns(1)),qx(1,ls));
// WRF:20274
chcnsh(1) = lake_min(chcns(1),lake_div((rho0(1)*qhcns(1)),((r*coeff_xvmn{}(lh)))));
// WRF:20276
vhcns(1) = lake_div((rho0(1)*qhcns(1)),r);
// WRF:20277
}
// WRF:20279
} else if((iglcnvs==3)) {
// WRF:20284
if((tmp>coeff_xdnmn{}(lh))) {
// WRF:20285
r = lake_max((0.5f*((xdn(1,ls)+tmp))),coeff_xdnmn{}(lh));
// WRF:20287
qhcns(1) = (0.5f*qsacw(1));
// WRF:20288
chcns(1) = lake_div(qhcns(1),((gamsnow73fac*xmas(1,ls))));
// WRF:20289
chcns(1) = lake_min(chcns(1),lake_div((cx(1,ls)*qhcns(1)),qx(1,ls)));
// WRF:20290
chcnsh(1) = lake_min(chcns(1),lake_div((rho0(1)*qhcns(1)),((r*coeff_xvmn{}(lh)))));
// WRF:20291
vhcns(1) = lake_div((rho0(1)*qhcns(1)),r);
// WRF:20292
}
// WRF:20294
}
// WRF:20296
}
// WRF:20298
}
// WRF:20301
}
// WRF:20303
} else {
// WRF:20305
qhcns(1) = ((0.001f*ehscnv(1))*lake_max(((qx(1,ls)-6.e-4f)),0.0f));
// WRF:20306
qhcns(1) = lake_min(qhcns(1),qxmxd(1,ls));
// WRF:20307
if((coeff_lvol{}(lh)>=1)) {vhcns(1) = lake_div((rho0(1)*qhcns(1)),lake_max(xdn(1,ls),400.f));}
// WRF:20309
}
// WRF:20310
}
// WRF:20317
if(((irwfrz>0)&&(!mixedphase))) {
// WRF:20319
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20323
qrztot(1) = ((qrfrz(1)+qiacr(1))+qsacr(1));
// WRF:20328
qrzmax(1) = ((((xdia(1,lr,1)*rwvent(1))*cx(1,lr))*fwet1(1)));
// WRF:20330
qrzmax(1) = lake_max(qrzmax(1),0.0f);
// WRF:20331
qrzmax(1) = lake_min(qrztot(1),qrzmax(1));
// WRF:20332
qrzmax(1) = lake_min((qx(1,lr)*dtpinv),qrzmax(1));
// WRF:20334
if((temcg(1)<(-30.f))) {
// WRF:20335
qrzmax(1) = (qx(1,lr)*dtpinv);
// WRF:20336
}
// WRF:20342
if(((qrztot(1)>qrzmax(1))&&(qrztot(1)>coeff_qxmin{}(lr)))) {
// WRF:20343
qrzfac(1) = lake_div(qrzmax(1),(qrztot(1)));
// WRF:20344
} else {
// WRF:20345
qrzfac(1) = 1.0f;
// WRF:20346
}
// WRF:20347
qrzfac(1) = lake_min(1.0f,qrzfac(1));
// WRF:20349
}
// WRF:20355
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20356
if(((temg(1)<=273.15f)&&(qrzfac(1)<1.0f))) {
// WRF:20357
qrfrz(1) = (qrzfac(1)*qrfrz(1));
// WRF:20358
qrfrzs(1) = (qrzfac(1)*qrfrzs(1));
// WRF:20359
qrfrzf(1) = (qrzfac(1)*qrfrzf(1));
// WRF:20360
qiacr(1) = (qrzfac(1)*qiacr(1));
// WRF:20361
qsacr(1) = (qrzfac(1)*qsacr(1));
// WRF:20362
qiacrf(1) = (qrzfac(1)*qiacrf(1));
// WRF:20363
qiacrs(1) = (qrzfac(1)*qiacrs(1));
// WRF:20364
crfrz(1) = (qrzfac(1)*crfrz(1));
// WRF:20365
crfrzf(1) = (qrzfac(1)*crfrzf(1));
// WRF:20366
crfrzs(1) = (qrzfac(1)*crfrzs(1));
// WRF:20367
ciacr(1) = (qrzfac(1)*ciacr(1));
// WRF:20368
ciacrf(1) = (qrzfac(1)*ciacrf(1));
// WRF:20369
ciacrs(1) = (qrzfac(1)*ciacrs(1));
// WRF:20376
vrfrzf(1) = (qrzfac(1)*vrfrzf(1));
// WRF:20377
viacrf(1) = (qrzfac(1)*viacrf(1));
// WRF:20378
}
// WRF:20379
}
// WRF:20383
}
// WRF:20391
for(int _s0=1;_s0<=ngs;++_s0) {qrcev(_s0) = 0.0f;}
// WRF:20392
for(int _s0=1;_s0<=ngs;++_s0) {crcev(_s0) = 0.0f;}
// WRF:20395
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20397
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:20399
qrcev(1) = ((((fvce(1)*cx(1,lr))*rwvent(1))*rwcap(1))*evapfac);
// WRF:20402
if((rcond==1)) {
// WRF:20403
qrcev(1) = lake_min(qrcev(1),qxmxd(1,lv));
// WRF:20405
} else {
// WRF:20406
qrcev(1) = lake_min(qrcev(1),0.0f);
// WRF:20407
}
// WRF:20409
qrcev(1) = lake_max(qrcev(1),(-qrmxd(1)));
// WRF:20411
if(((qrcev(1)<0.f)&&(lnr>1))) {
// WRF:20414
if((icrcev==1)) {
// WRF:20415
crcev(1) = ((lake_div(cx(1,lr),(qx(1,lr))))*qrcev(1));
// WRF:20416
} else if((icrcev==2)) {
// WRF:20417
crcev(1) = lake_div((((lake_div(cx(1,lr),(qx(1,lr))))*qrcev(1))*vtxbar(1,lr,2)),vtxbar(1,lr,1));
// WRF:20418
} else {
// WRF:20419
crcev(1) = 0.0f;
// WRF:20420
}
// WRF:20421
} else {
// WRF:20422
crcev(1) = 0.0f;
// WRF:20423
}
// WRF:20426
}
// WRF:20428
}
// WRF:20432
if((lhwlg>1)) {
// WRF:20433
for(int _s0=1;_s0<=ngs;++_s0) {qhcevlg(_s0) = 0.0f;}
// WRF:20434
for(int _s0=1;_s0<=ngs;++_s0) {chcevlg(_s0) = 0.0f;}
// WRF:20435
}
// WRF:20436
if((lhlwlg>1)) {
// WRF:20437
for(int _s0=1;_s0<=ngs;++_s0) {qhlcevlg(_s0) = 0.0f;}
// WRF:20438
for(int _s0=1;_s0<=ngs;++_s0) {chlcevlg(_s0) = 0.0f;}
// WRF:20439
}
// WRF:20449
for(int _s0=1;_s0<=ngs;++_s0) {chmul1(_s0) = 0.0f;}
// WRF:20450
for(int _s0=1;_s0<=ngs;++_s0) {chlmul1(_s0) = 0.0f;}
// WRF:20451
for(int _s0=1;_s0<=ngs;++_s0) {csmul1(_s0) = 0.0f;}
// WRF:20453
for(int _s0=1;_s0<=ngs;++_s0) {qhmul1(_s0) = 0.0f;}
// WRF:20454
for(int _s0=1;_s0<=ngs;++_s0) {qhlmul1(_s0) = 0.0f;}
// WRF:20455
for(int _s0=1;_s0<=ngs;++_s0) {qsmul1(_s0) = 0.0f;}
// WRF:20456
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20458
ltest = (qx(1,lh)>coeff_qxmin{}(lh));
// WRF:20459
if((lhl>1)) {ltest = (ltest||(qx(1,lhl)>coeff_qxmin{}(lhl)));}
// WRF:20461
if(((((itype1>=1)||(itype2>=1)))&&(qx(1,lc)>coeff_qxmin{}(lc)))) {
// WRF:20463
if(((temg(1)>=265.15f)&&(temg(1)<=271.15f))) {
// WRF:20464
if((ipconc>=2)) {
// WRF:20465
if(((xv(1,lc)>0.0f)&&ltest)) {
// WRF:20472
if((alpha(1,lc)==0.0f)) {
// WRF:20473
ex1 = ((lake_div(1.f,250.f))*gfk_exp(lake_div((-7.23e-15f),xv(1,lc))));
// WRF:20474
} else {
// WRF:20476
ratio = lake_div((((1.f+alpha(1,lc)))*(7.23e-15f)),xv(1,lc));
// WRF:20478
if(usegamxinfcnu) {
// WRF:20479
i = ((int)round((dgami*((1.f+alpha(1,lc))))));
// WRF:20480
gcnup1 = gmoi(i);
// WRF:20481
ex1 = lake_div(((lake_div(1.f,250.f))*gamxinf((1.f+alpha(1,lc)),ratio)),(gcnup1));
// WRF:20482
} else {
// WRF:20483
ratio = lake_min(maxratiolu,ratio);
// WRF:20484
tmp = gaminterp(ratio,alpha(1,lc),1,1);
// WRF:20485
ex1 = ((lake_div(1.f,250.f))*tmp);
// WRF:20486
}
// WRF:20487
}
// WRF:20488
if((itype2<=2)) {
// WRF:20489
ft = lake_max(0.0f,lake_min(1.0f,((((-0.11f)*lake_pow(temcg(1),2))-(1.1f*temcg(1)))-1.7f)));
// WRF:20490
} else {
// WRF:20491
if(((temg(1)>=265.15f)&&(temg(1)<=267.15f))) {
// WRF:20492
ft = 0.5f;
// WRF:20493
} else if(((temg(1)>=267.15f)&&(temg(1)<=269.15f))) {
// WRF:20494
ft = 1.0f;
// WRF:20495
} else if(((temg(1)>=269.15f)&&(temg(1)<=271.15f))) {
// WRF:20496
ft = 0.5f;
// WRF:20497
} else {
// WRF:20498
ft = 0.0f;
// WRF:20499
}
// WRF:20500
}
// WRF:20504
if((ft>0.0f)) {
// WRF:20506
if((itype2>0)) {
// WRF:20507
if(((qx(1,lh)>coeff_qxmin{}(lh))&&((!wetsfc(1))))) {
// WRF:20508
chmul1(1) = ((ft*ex1)*chacw(1));
// WRF:20510
qhmul1(1) = ((cimas0*chmul1(1))*rhoinv(1));
// WRF:20511
}
// WRF:20512
if((lhl>1)) {
// WRF:20513
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&((!wetsfchl(1))))) {
// WRF:20514
chlmul1(1) = (((ft*ex1)*chlacw(1)));
// WRF:20515
qhlmul1(1) = ((cimas0*chlmul1(1))*rhoinv(1));
// WRF:20516
}
// WRF:20517
}
// WRF:20518
}
// WRF:20520
if((itype1>0)) {
// WRF:20521
if(((qx(1,lh)>coeff_qxmin{}(lh))&&((!wetsfc(1))))) {
// WRF:20522
tmp = (((ft*(3.5e+08f))*rho0(1))*qhacw(1));
// WRF:20523
chmul1(1) = (chmul1(1)+tmp);
// WRF:20524
qhmul1(1) = (qhmul1(1)+((cimas0*tmp)*rhoinv(1)));
// WRF:20525
}
// WRF:20526
if((lhl>1)) {
// WRF:20527
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&((!wetsfchl(1))))) {
// WRF:20528
tmp = (((ft*(3.5e+08f))*rho0(1))*qhlacw(1));
// WRF:20529
chlmul1(1) = (chlmul1(1)+tmp);
// WRF:20530
qhlmul1(1) = (qhlmul1(1)+((cimas0*tmp)*rhoinv(1)));
// WRF:20531
}
// WRF:20532
}
// WRF:20533
}
// WRF:20536
}
// WRF:20538
}
// WRF:20540
} else {
// WRF:20544
fimt1(1) = 0.0f;
// WRF:20548
if(((temg(1)>=268.15f)&&(temg(1)<=270.15f))) {
// WRF:20549
fimt1(1) = (1.0f-lake_div(((temg(1)-268.15f)),2.0f));
// WRF:20550
} else if(((temg(1)<=268.15f)&&(temg(1)>=265.15f))) {
// WRF:20551
fimt1(1) = (1.0f+lake_div(((temg(1)-268.15f)),3.0f));
// WRF:20552
} else {
// WRF:20553
fimt1(1) = 0.0f;
// WRF:20554
}
// WRF:20558
if(((temg(1)>=265.15f)&&(temg(1)<=267.15f))) {
// WRF:20559
fimt1(1) = 0.5f;
// WRF:20560
} else if(((temg(1)>=267.15f)&&(temg(1)<=269.15f))) {
// WRF:20561
fimt1(1) = 1.0f;
// WRF:20562
} else if(((temg(1)>=269.15f)&&(temg(1)<=271.15f))) {
// WRF:20563
fimt1(1) = 0.5f;
// WRF:20564
} else {
// WRF:20565
fimt1(1) = 0.0f;
// WRF:20566
}
// WRF:20573
if((itype1>=1)) {
// WRF:20574
fimta(1) = ((3.5e+08f)*rho0(1));
// WRF:20575
} else {
// WRF:20576
fimta(1) = 0.0f;
// WRF:20577
}
// WRF:20586
fimt2(1) = 0.0f;
// WRF:20587
xcwmas = (xmas(1,lc)*1000.f);
// WRF:20589
if((itype2>=1)) {
// WRF:20590
if((xcwmas<1.26e-9f)) {
// WRF:20591
fimt2(1) = 0.0f;
// WRF:20592
}
// WRF:20593
if(((xcwmas<=3.55e-9f)&&(xcwmas>=1.26e-9f))) {
// WRF:20594
fimt2(1) = (((2.27f)*gfk_log(xcwmas))+13.39f);
// WRF:20595
}
// WRF:20596
if((xcwmas>3.55e-9f)) {
// WRF:20597
fimt2(1) = 1.0f;
// WRF:20598
}
// WRF:20600
fimt2(1) = lake_min(fimt2(1),1.0f);
// WRF:20601
fimt2(1) = lake_max(fimt2(1),0.0f);
// WRF:20603
}
// WRF:20615
if((!wetsfc(1))) {
// WRF:20616
chmul1(1) = ((fimt1(1)*((fimta(1)+((4.0e-03f)*fimt2(1)))))*qhacw(1));
// WRF:20618
}
// WRF:20620
qhmul1(1) = (chmul1(1)*(lake_div(cimas0,rho0(1))));
// WRF:20622
if((lhl>1)) {
// WRF:20623
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&((!wetsfchl(1))))) {
// WRF:20624
tmp = ((fimt1(1)*((fimta(1)+((4.0e-03f)*fimt2(1)))))*qhlacw(1));
// WRF:20626
chlmul1(1) = tmp;
// WRF:20627
qhlmul1(1) = ((cimas0*tmp)*rhoinv(1));
// WRF:20628
}
// WRF:20629
}
// WRF:20633
}
// WRF:20635
}
// WRF:20637
}
// WRF:20639
}
// WRF:20652
for(int _s0=1;_s0<=ngs;++_s0) {csmul(_s0) = 0.0f;}
// WRF:20653
for(int _s0=1;_s0<=ngs;++_s0) {qsmul(_s0) = 0.0f;}
// WRF:20655
if((isnwfrac!=0)) {
// WRF:20656
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20657
if((temg(1)>265.0f)) {
// WRF:20658
if(((xdia(1,ls,1)>100.e-6f)&&(xdia(1,ls,1)<2.0e-3f))) {
// WRF:20660
tmp = ((((rhoinv(1)*pi)*xdn(1,ls))*cx(1,ls))*lake_pow((500.e-6f),3));
// WRF:20661
qsmul(1) = lake_max((kfrag*((qx(1,ls)-tmp))),0.0f);
// WRF:20663
qsmul(1) = lake_min(qxmxd(1,li),qsmul(1));
// WRF:20664
csmul(1) = lake_min(cxmxd(1,li),lake_div((rho0(1)*qsmul(1)),mfrag));
// WRF:20666
}
// WRF:20667
}
// WRF:20668
}
// WRF:20669
}
// WRF:20680
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20681
qracif(1) = qraci(1);
// WRF:20682
cracif(1) = craci(1);
// WRF:20684
}
// WRF:20708
cmassin = cimasn;
// WRF:20709
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20710
qiint(1) = 0.0f;
// WRF:20711
ciint(1) = 0.0f;
// WRF:20712
qicicnt(1) = 0.0f;
// WRF:20713
cicint(1) = 0.0f;
// WRF:20714
qipipnt(1) = 0.0f;
// WRF:20715
cipint(1) = 0.0f;
// WRF:20716
ccitmp = 0.0f;
// WRF:20717
if((((icenucopt==1)||(icenucopt==(-10)))||(icenucopt==(-11)))) {
// WRF:20718
if(((((temg(1)<268.15f)||(((imeyers5&&(temg(1)<272.0f))&&(temgkm2(1)<tfr)))))&&(ciintmx>((cx(1,li)+ccitmp))))) {
// WRF:20724
fiinit(1) = lake_div((lake_pow(felv(1),2)),((cp*rw)));
// WRF:20725
dqisdt(1) = lake_div(((qx(1,lv)-qis(1))),((1.0f+lake_div((fiinit(1)*qis(1)),tsqr(1)))));
// WRF:20728
idqis = 0;
// WRF:20729
if((ssi(1)>1.0f)) {
// WRF:20730
idqis = 1;
// WRF:20731
dzfacp = float(lake_max(lake_min(global_k+1,global_nz-1)-global_k,0));
// WRF:20732
dzfacm = float(lake_max(global_k-lake_max(global_k-1,1),0));
// WRF:20733
qiint(1) = lake_div(lake_div(((((idqis*il5(1))*(lake_div(cmassin,rho0(1))))*lake_max(0.0f,wvel(1)))*lake_max(((cninp(1)-cninm(1))),0.0f)),gz(1,jgs,1)),(((dzfacp+dzfacm))));
// WRF:20740
qiint(1) = lake_min(qiint(1),lake_max((0.25f*dqisdt(1)),0.0f));
// WRF:20741
ciint(1) = lake_div((qiint(1)*rho0(1)),cmassin);
// WRF:20749
if((icenucopt!=(-10))) {
// WRF:20751
if((lcin>1)) {
// WRF:20752
ciint(1) = lake_min(ciint(1),(ccin(1)*dtpinv));
// WRF:20753
ccin(1) = (ccin(1)-(ciint(1)*dtp));
// WRF:20754
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20755
} else if((lcina>1)) {
// WRF:20756
ciint(1) = lake_max(0.0f,lake_min(ciint(1),(lake_min(cnina(1),ciintmx)-cina(1))));
// WRF:20757
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20759
} else if(((icenucopt==1)&&(ciint(1)>(lake_max(0.0f,((ciintmx-cx(1,li))-ccitmp))*dtpinv)))) {
// WRF:20760
ciint(1) = (lake_max(0.0f,(ciintmx-(cx(1,li))))*dtpinv);
// WRF:20761
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20763
} else if(((icenucopt==(-11))&&((dtp*ciint(1))>((cnina(1)-((cx(1,li)-ccitmp))))))) {
// WRF:20764
ciint(1) = lake_max(0.0f,(cnina(1)-(((cx(1,li)+ccitmp))*dtpinv)));
// WRF:20765
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20767
}
// WRF:20768
}
// WRF:20770
}
// WRF:20771
}
// WRF:20773
} else if((((icenucopt==2)||(icenucopt==(-1)))||(icenucopt==(-2)))) {
// WRF:20775
if(((((temg(1)<268.15f)&&(ssw(1)>1.0f)))||(ssi(1)>1.25f))) {
// WRF:20776
if((lcin>1)) {
// WRF:20777
ciint(1) = lake_min(cnina(1),ccin(1));
// WRF:20778
ciint(1) = lake_min(ciint(1),lake_max(0.0f,(ciintmx-((cx(1,li)-ccitmp)))));
// WRF:20779
ccin(1) = (ccin(1)-ciint(1));
// WRF:20780
ciint(1) = (ciint(1)*dtpinv);
// WRF:20781
} else {
// WRF:20782
ciint(1) = (lake_max(0.0f,(cnina(1)-cina(1)))*dtpinv);
// WRF:20783
}
// WRF:20784
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20786
fiinit(1) = lake_div((lake_pow(felv(1),2)),((cp*rw)));
// WRF:20787
dqisdt(1) = lake_div(((qx(1,lv)-qis(1))),((1.0f+lake_div((fiinit(1)*qis(1)),tsqr(1)))));
// WRF:20788
qiint(1) = lake_min(qiint(1),lake_max((0.25f*dqisdt(1)),0.0f));
// WRF:20789
ciint(1) = lake_div((qiint(1)*rho0(1)),cmassin);
// WRF:20790
}
// WRF:20794
} else if((((icenucopt==3)||(icenucopt==4))||(icenucopt==10))) {
// WRF:20795
if((temg(1)<268.15f)) {
// WRF:20796
if((lcin>1)) {
// WRF:20797
ciint(1) = lake_min(cnina(1),ccin(1));
// WRF:20798
ciint(1) = lake_min(ciint(1),lake_max(0.0f,(ciintmx-((cx(1,li)+ccitmp)))));
// WRF:20799
ccin(1) = (ccin(1)-ciint(1));
// WRF:20800
ciint(1) = (ciint(1)*dtpinv);
// WRF:20801
} else {
// WRF:20802
ciint(1) = (lake_max(0.0f,(cnina(1)-cina(1)))*dtpinv);
// WRF:20803
}
// WRF:20804
qiint(1) = lake_div((ciint(1)*cmassin),rho0(1));
// WRF:20805
}
// WRF:20807
}
// WRF:20809
if((xplate(1)==1)) {
// WRF:20810
qipipnt(1) = qiint(1);
// WRF:20811
cipint(1) = ciint(1);
// WRF:20812
}
// WRF:20814
if((xcolmn(1)==1)) {
// WRF:20815
qicicnt(1) = qiint(1);
// WRF:20816
cicint(1) = ciint(1);
// WRF:20817
}
// WRF:20822
}
// WRF:20829
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:20832
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:20851
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20852
qrshr(1) = 0.0f;
// WRF:20853
qwshw(1) = 0.0f;
// WRF:20854
cwshw(1) = 0.0f;
// WRF:20855
qsshrp(1) = 0.0f;
// WRF:20856
qhshrp(1) = 0.0f;
// WRF:20857
}
// WRF:20863
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20864
qrshr(1) = ((qsshr(1)+qhshr(1))+qhlshr(1));
// WRF:20865
crshr(1) = (lake_div(chshrr(1),rzxh(1))+lake_div(chlshrr(1),rzxhl(1)));
// WRF:20868
if((ipconc>=3)) {
// WRF:20870
}
// WRF:20871
}
// WRF:20880
if((ipconc>=1)) {
// WRF:20889
for(int _s0=1;_s0<=ngs;++_s0) {pccwi(_s0) = 0.0f;}
// WRF:20890
for(int _s0=1;_s0<=ngs;++_s0) {pccwd(_s0) = 0.0f;}
// WRF:20891
for(int _s0=1;_s0<=ngs;++_s0) {pccwdacc(_s0) = 0.0f;}
// WRF:20892
for(int _s0=1;_s0<=ngs;++_s0) {pccii(_s0) = 0.0f;}
// WRF:20893
for(int _s0=1;_s0<=ngs;++_s0) {pccin(_s0) = 0.0f;}
// WRF:20894
for(int _s0=1;_s0<=ngs;++_s0) {pccid(_s0) = 0.0f;}
// WRF:20895
for(int _s0=1;_s0<=ngs;++_s0) {pcisi(_s0) = 0.0f;}
// WRF:20896
for(int _s0=1;_s0<=ngs;++_s0) {pcisd(_s0) = 0.0f;}
// WRF:20897
for(int _s0=1;_s0<=ngs;++_s0) {pcrwi(_s0) = 0.0f;}
// WRF:20898
for(int _s0=1;_s0<=ngs;++_s0) {pcrwd(_s0) = 0.0f;}
// WRF:20899
for(int _s0=1;_s0<=ngs;++_s0) {pcswi(_s0) = 0.0f;}
// WRF:20900
for(int _s0=1;_s0<=ngs;++_s0) {pcswd(_s0) = 0.0f;}
// WRF:20901
for(int _s0=1;_s0<=ngs;++_s0) {pchwi(_s0) = 0.0f;}
// WRF:20902
for(int _s0=1;_s0<=ngs;++_s0) {pchwd(_s0) = 0.0f;}
// WRF:20903
for(int _s0=1;_s0<=ngs;++_s0) {pchli(_s0) = 0.0f;}
// WRF:20904
for(int _s0=1;_s0<=ngs;++_s0) {pchld(_s0) = 0.0f;}
// WRF:20911
if((warmonly<0.5f)) {
// WRF:20912
if((ffrzs<1.0f)) {
// WRF:20913
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20914
pccii(1) = (((((((il5(1)*cicint(1))+(il5(1)*((((((1.0f-cwfrz2snowfrac))*cwfrzc(1))+cwctfzc(1))+cicichr(1)))))+chmul1(1))+chlmul1(1))+csplinter(1))+csplinter2(1))+csmul(1));
// WRF:20923
pccii(1) = (pccii(1)*((1.0f-ffrzs)));
// WRF:20926
pccid(1) = (((il5(1)*((((((((-cscni(1))-cscnvi(1))-craci(1))-csaci(1))-chaci(1))-chlaci(1))-chcni(1))))+(il5(1)*cisbv(1)))-(((1.f-il5(1)))*cimlr(1)));
// WRF:20935
pccin(1) = ciint(1);
// WRF:20938
}
// WRF:20939
}
// WRF:20940
} else if((warmonly<0.8f)) {
// WRF:20941
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20947
pccii(1) = (((((((il5(1)*cicint(1))+(il5(1)*((((((1.0f-cwfrz2snowfrac))*cwfrzc(1))+cwctfzc(1))+cicichr(1)))))+chmul1(1))+chlmul1(1))+csplinter(1))+csplinter2(1))+csmul(1));
// WRF:20956
pccii(1) = (pccii(1)*((1.f-ffrzs)));
// WRF:20957
pccid(1) = (((+il5(1))*cisbv(1))-(((1.f-il5(1)))*cimlr(1)));
// WRF:20966
pccin(1) = ciint(1);
// WRF:20968
}
// WRF:20969
}
// WRF:20976
if((ipconc>=2)) {
// WRF:20978
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:20979
pccwi(1) = ((0.0f)-cwshw(1));
// WRF:20981
if((warmonly<0.5f)) {
// WRF:20982
pccwd(1) = ((((((-cautn(1))+(il5(1)*(((((-ciacw(1))-cwfrz(1))-cwctfzp(1))-cwctfzc(1)))))-cracw(1))-csacw(1))-chacw(1))-chlacw(1));
// WRF:20990
} else if((warmonly<0.8f)) {
// WRF:20991
pccwd(1) = (((((-cautn(1))+(il5(1)*(((((-ciacw(1))-cwfrz(1))-cwctfzp(1))-cwctfzc(1)))))-cracw(1))-chacw(1))-chlacw(1));
// WRF:20998
} else {
// WRF:21013
pccwd(1) = ((-cautn(1))-cracw(1));
// WRF:21015
}
// WRF:21018
if(((false&&(exwmindiam>0.0f))&&(ccwresv(1)>0.0f))) {
// WRF:21019
pccwdacc(1) = (((((il5(1)*((-ciacw(1))))-cracw(1))-csacw(1))-chacw(1))-chlacw(1));
// WRF:21024
if((((-pccwdacc(1))*dtp)>(cx(1,lc)-ccwresv(1)))) {
// WRF:21026
frac = lake_div((-((cx(1,lc)-ccwresv(1)))),((pccwdacc(1)*dtp)));
// WRF:21027
pccwdacc(1) = ((-((cx(1,lc)-ccwresv(1))))*dtpinv);
// WRF:21029
ciacw(1) = (frac*ciacw(1));
// WRF:21030
cracw(1) = (frac*cracw(1));
// WRF:21031
csacw(1) = (frac*csacw(1));
// WRF:21032
chacw(1) = (frac*chacw(1));
// WRF:21033
cautn(1) = (frac*cautn(1));
// WRF:21035
if((lhl>1)) {chlacw(1) = (frac*chlacw(1));}
// WRF:21038
pccwd(1) = ((((((-cautn(1))+(il5(1)*(((((((-ciacw(1))-cwfrzp(1))-cwctfzp(1))-cwfrzc(1))-cwctfzc(1))-(il5(1)*(ciihr(1)))))))-cracw(1))-csacw(1))-chacw(1))-chlacw(1));
// WRF:21046
}
// WRF:21048
}
// WRF:21051
if((((-pccwd(1))*dtp)>cx(1,lc))) {
// WRF:21058
frac = lake_div((-cx(1,lc)),((pccwd(1)*dtp)));
// WRF:21059
pccwd(1) = ((-cx(1,lc))*dtpinv);
// WRF:21061
ciacw(1) = (frac*ciacw(1));
// WRF:21062
cwfrz(1) = (frac*cwfrz(1));
// WRF:21063
cwfrzp(1) = (frac*cwfrzp(1));
// WRF:21064
cwctfzp(1) = (frac*cwctfzp(1));
// WRF:21065
cwfrzc(1) = (frac*cwfrzc(1));
// WRF:21066
cwctfzc(1) = (frac*cwctfzc(1));
// WRF:21067
cwctfz(1) = (frac*cwctfz(1));
// WRF:21068
cracw(1) = (frac*cracw(1));
// WRF:21069
csacw(1) = (frac*csacw(1));
// WRF:21070
chacw(1) = (frac*chacw(1));
// WRF:21071
cautn(1) = (frac*cautn(1));
// WRF:21073
pccii(1) = (pccii(1)-(((((1.f-frac))*il5(1))*((cwfrzc(1)+cwctfzc(1))))*((1.f-ffrzs))));
// WRF:21074
if((lhl>1)) {chlacw(1) = (frac*chlacw(1));}
// WRF:21077
}
// WRF:21079
}
// WRF:21081
}
// WRF:21086
if((ipconc>=3)) {
// WRF:21088
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21090
if((warmonly<0.5f)) {
// WRF:21091
pcrwi(1) = ((crcnw(1)+(((1-il5(1)))*((((lake_div((-chmlrr(1)),rzxh(1))-lake_div(chlmlrr(1),rzxhl(1)))-csmlrr(1))-cimlr(1)))))-crshr(1));
// WRF:21101
pcrwd(1) = (((((il5(1)*(((-ciacr(1))-crfrz(1))))-chacr(1))-chlacr(1))+crcev(1))-cracr(1));
// WRF:21110
} else if((warmonly<0.8f)) {
// WRF:21111
pcrwi(1) = ((crcnw(1)+(((1-il5(1)))*((((lake_div((-chmlrr(1)),rzxh(1))-lake_div(chlmlrr(1),rzxhl(1)))-csmlrr(1))-cimlr(1)))))-crshr(1));
// WRF:21120
pcrwd(1) = (((((il5(1)*((-crfrz(1))))-chacr(1))-chlacr(1))+crcev(1))-cracr(1));
// WRF:21126
} else {
// WRF:21127
pcrwi(1) = crcnw(1);
// WRF:21129
pcrwd(1) = ((+crcev(1))-cracr(1));
// WRF:21138
}
// WRF:21141
frac = 0.0f;
// WRF:21142
if((((-pcrwd(1))*dtp)>cx(1,lr))) {
// WRF:21150
frac = lake_div((-cx(1,lr)),((pcrwd(1)*dtp)));
// WRF:21151
pcrwd(1) = ((-cx(1,lr))*dtpinv);
// WRF:21153
ciacr(1) = (frac*ciacr(1));
// WRF:21154
ciacrf(1) = (frac*ciacrf(1));
// WRF:21155
ciacrs(1) = (frac*ciacrs(1));
// WRF:21156
crfrz(1) = (frac*crfrz(1));
// WRF:21157
crfrzf(1) = (frac*crfrzf(1));
// WRF:21158
crfrzs(1) = (frac*crfrzs(1));
// WRF:21159
chacr(1) = (frac*chacr(1));
// WRF:21160
chlacr(1) = (frac*chlacr(1));
// WRF:21161
crcev(1) = (frac*crcev(1));
// WRF:21162
cracr(1) = (frac*cracr(1));
// WRF:21165
}
// WRF:21167
}
// WRF:21169
}
// WRF:21172
if((warmonly<0.5f)) {
// WRF:21177
if((ipconc>=4)) {
// WRF:21179
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21180
pcswi(1) = (((il5(1)*((cscnis(1)+cscnvis(1))))+lake_div((cwfrz2snowfrac*cwfrz(1)),cwfrz2snowratio))+cscnh(1));
// WRF:21185
if((ffrzs>0.0f)) {
// WRF:21186
pcswi(1) = (pcswi(1)+(ffrzs*((((((((il5(1)*cicint(1))+(il5(1)*(((cwfrzc(1)+cwctfzc(1))+cicichr(1)))))+chmul1(1))+chlmul1(1))+csplinter(1))+csplinter2(1))+csmul(1)))));
// WRF:21194
}
// WRF:21197
if((ess0<0.0f)) {
// WRF:21198
csacs(1) = lake_max(0.0f,(csacs(1)-((ifrzs)*((crfrzs(1)+ciacrs(1))))));
// WRF:21199
}
// WRF:21201
pcswd(1) = (((((((-chacs(1))-chlacs(1))-chcns(1))+(((1-il5(1)))*csmlr(1)))+csshr(1))+cssbv(1))-csacs(1));
// WRF:21210
frac = 0.0f;
// WRF:21211
if((imixedphase==0)) {
// WRF:21212
if(((cx(1,ls)+(dtp*((pcswi(1)+pcswd(1)))))<0.0f)) {
// WRF:21213
frac = lake_div((((-cx(1,ls))+(pcswi(1)*dtp))),((pcswd(1)*dtp)));
// WRF:21215
pcswd(1) = (frac*pcswd(1));
// WRF:21217
chacs(1) = (frac*chacs(1));
// WRF:21218
chlacs(1) = (frac*chlacs(1));
// WRF:21219
chcns(1) = (frac*chcns(1));
// WRF:21220
csmlr(1) = (frac*csmlr(1));
// WRF:21221
csshr(1) = (frac*csshr(1));
// WRF:21222
cssbv(1) = (frac*cssbv(1));
// WRF:21223
csacs(1) = (frac*csacs(1));
// WRF:21225
}
// WRF:21226
}
// WRF:21230
pccii(1) = ((pccii(1)+(((1.f-ifrzs))*crfrzs(1)))+(((1.f-ifrzs))*ciacrs(1)));
// WRF:21234
pcswi(1) = ((pcswi(1)+((ifrzs)*crfrzs(1)))+((ifrzs)*ciacrs(1)));
// WRF:21238
}
// WRF:21240
}
// WRF:21245
if((ipconc>=5)) {
// WRF:21246
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21247
pchwi(1) = ((((+((((ffrzh*ifrzg)*crfrzf(1))+(((il5(1)*ffrzh)*ifiacrg)*(ciacrf(1))))))+(f2h*chcnsh(1)))+(f2h*chcnih(1)))+chcnhl(1));
// WRF:21252
pchwd(1) = ((((((1-il5(1)))*chmlr(1))+chsbv(1))-(il5(1)*chlcnh(1)))-cscnh(1));
// WRF:21259
}
// WRF:21268
if(((lhl>1)&&(lnhl>1))) {
// WRF:21269
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21270
pchli(1) = (((((ffrzh*((1.0f-ifrzg)))*crfrzf(1))+(((il5(1)*ffrzh)*((1.0f-ifiacrg)))*(ciacrf(1)))))+(chlcnhhl(1)*rzxhlh(1)));
// WRF:21273
pchld(1) = (((((1-il5(1)))*chlmlr(1))+chlsbv(1))-chcnhl(1));
// WRF:21278
if((imixedphase==0)) {
// WRF:21279
frac = 0.0f;
// WRF:21280
if(((cx(1,lhl)+(dtp*((pchli(1)+pchld(1)))))<0.0f)) {
// WRF:21283
frac = lake_div((((-cx(1,lhl))+(pchli(1)*dtp))),((pchld(1)*dtp)));
// WRF:21285
chlmlr(1) = (frac*chlmlr(1));
// WRF:21286
chlsbv(1) = (frac*chlsbv(1));
// WRF:21287
chcnhl(1) = (frac*chcnhl(1));
// WRF:21289
pchld(1) = (frac*pchld(1));
// WRF:21291
}
// WRF:21292
}
// WRF:21294
}
// WRF:21296
}
// WRF:21299
}
// WRF:21301
} else if((warmonly<0.8f)) {
// WRF:21306
if((ipconc>=5)) {
// WRF:21307
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21308
pchwi(1) = ((+ifrzg)*(crfrzf(1)));
// WRF:21311
pchwd(1) = ((((1-il5(1)))*chmlr(1))-(il5(1)*chlcnh(1)));
// WRF:21314
}
// WRF:21318
if((lhl>1)) {
// WRF:21319
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21320
pchli(1) = ((((1.0f-ifrzg))*(crfrzf(1)))+lake_div((chlcnhhl(1)*rzxhl(1)),rzxh(1)));
// WRF:21323
pchld(1) = (((1-il5(1)))*chlmlr(1));
// WRF:21331
}
// WRF:21333
}
// WRF:21335
}
// WRF:21337
}
// WRF:21344
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21345
pctot(1) = (((((((((((pccwi(1)+pccwd(1))+pccii(1))+pccid(1))+pcrwi(1))+pcrwd(1))+pcswi(1))+pcswd(1))+pchwi(1))+pchwd(1))+pchli(1))+pchld(1));
// WRF:21351
}
// WRF:21354
}
// WRF:21364
for(int _s0=1;_s0<=ngs;++_s0) {pqwvi(_s0) = 0.0f;}
// WRF:21365
for(int _s0=1;_s0<=ngs;++_s0) {pqwvd(_s0) = 0.0f;}
// WRF:21366
for(int _s0=1;_s0<=ngs;++_s0) {pqcwi(_s0) = 0.0f;}
// WRF:21367
for(int _s0=1;_s0<=ngs;++_s0) {pqcwd(_s0) = 0.0f;}
// WRF:21368
for(int _s0=1;_s0<=ngs;++_s0) {pqcwdacc(_s0) = 0.0f;}
// WRF:21369
for(int _s0=1;_s0<=ngs;++_s0) {pqcii(_s0) = 0.0f;}
// WRF:21370
for(int _s0=1;_s0<=ngs;++_s0) {pqcid(_s0) = 0.0f;}
// WRF:21371
for(int _s0=1;_s0<=ngs;++_s0) {pqrwi(_s0) = 0.0f;}
// WRF:21372
for(int _s0=1;_s0<=ngs;++_s0) {pqrwd(_s0) = 0.0f;}
// WRF:21373
for(int _s0=1;_s0<=ngs;++_s0) {pqswi(_s0) = 0.0f;}
// WRF:21374
for(int _s0=1;_s0<=ngs;++_s0) {pqswd(_s0) = 0.0f;}
// WRF:21375
for(int _s0=1;_s0<=ngs;++_s0) {pqhwi(_s0) = 0.0f;}
// WRF:21376
for(int _s0=1;_s0<=ngs;++_s0) {pqhwd(_s0) = 0.0f;}
// WRF:21377
for(int _s0=1;_s0<=ngs;++_s0) {pqhli(_s0) = 0.0f;}
// WRF:21378
for(int _s0=1;_s0<=ngs;++_s0) {pqhld(_s0) = 0.0f;}
// WRF:21379
for(int _s0=1;_s0<=ngs;++_s0) {pqlwsi(_s0) = 0.0f;}
// WRF:21380
for(int _s0=1;_s0<=ngs;++_s0) {pqlwsd(_s0) = 0.0f;}
// WRF:21381
for(int _s0=1;_s0<=ngs;++_s0) {pqlwhi(_s0) = 0.0f;}
// WRF:21382
for(int _s0=1;_s0<=ngs;++_s0) {pqlwhd(_s0) = 0.0f;}
// WRF:21383
for(int _s0=1;_s0<=ngs;++_s0) {pqlwlghi(_s0) = 0.0f;}
// WRF:21384
for(int _s0=1;_s0<=ngs;++_s0) {pqlwlghd(_s0) = 0.0f;}
// WRF:21385
for(int _s0=1;_s0<=ngs;++_s0) {pqlwlghli(_s0) = 0.0f;}
// WRF:21386
for(int _s0=1;_s0<=ngs;++_s0) {pqlwlghld(_s0) = 0.0f;}
// WRF:21387
for(int _s0=1;_s0<=ngs;++_s0) {pqlwhli(_s0) = 0.0f;}
// WRF:21388
for(int _s0=1;_s0<=ngs;++_s0) {pqlwhld(_s0) = 0.0f;}
// WRF:21389
if((ipconc>5)) {
// WRF:21390
for(int _s0=1;_s0<=ngs;++_s0) {pzhwi(_s0) = 0.0f;}
// WRF:21391
for(int _s0=1;_s0<=ngs;++_s0) {pzhwd(_s0) = 0.0f;}
// WRF:21392
for(int _s0=1;_s0<=ngs;++_s0) {pzrwi(_s0) = 0.0f;}
// WRF:21393
for(int _s0=1;_s0<=ngs;++_s0) {pzrwd(_s0) = 0.0f;}
// WRF:21394
for(int _s0=1;_s0<=ngs;++_s0) {pzhli(_s0) = 0.0f;}
// WRF:21395
for(int _s0=1;_s0<=ngs;++_s0) {pzhld(_s0) = 0.0f;}
// WRF:21396
}
// WRF:21402
if((warmonly<0.5f)) {
// WRF:21403
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21406
pqwvi(1) = ((((((((-lake_min(0.0f,qrcev(1)))-lake_min(0.0f,qhcev(1)))-lake_min(0.0f,qhlcev(1)))-lake_min(0.0f,qscev(1)))-qhsbv(1))-qhlsbv(1))-qssbv(1))-(il5(1)*qisbv(1)));
// WRF:21416
pqwvd(1) = ((((((-lake_max(0.0f,qrcev(1)))-lake_max(0.0f,qhcev(1)))-lake_max(0.0f,qhlcev(1)))-lake_max(0.0f,qscev(1)))+(il5(1)*(((((-qiint(1))-qhdpv(1))-qsdpv(1))-qhldpv(1)))))-(il5(1)*qidpv(1)));
// WRF:21425
}
// WRF:21427
} else if((warmonly<0.8f)) {
// WRF:21428
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21429
pqwvi(1) = ((-lake_min(0.0f,qrcev(1)))-(il5(1)*qisbv(1)));
// WRF:21432
pqwvd(1) = ((((+il5(1))*((((-qiint(1))-qhdpv(1))-qhldpv(1))))-lake_max(0.0f,qrcev(1)))-(il5(1)*qidpv(1)));
// WRF:21439
}
// WRF:21441
} else {
// WRF:21442
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21443
pqwvi(1) = (-lake_min(0.0f,qrcev(1)));
// WRF:21445
pqwvd(1) = (-lake_max(0.0f,qrcev(1)));
// WRF:21447
}
// WRF:21449
}
// WRF:21453
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21455
pqcwi(1) = (((0.0f)+qwcnr(1))-qwshw(1));
// WRF:21457
if((warmonly<0.5f)) {
// WRF:21458
pqcwd(1) = (((((((il5(1)*((((-qiacw(1))-qwfrz(1))-qwctfz(1))))-(il5(1)*(qiihr(1))))-qracw(1))-qsacw(1))-qrcnw(1))-qhacw(1))-qhlacw(1));
// WRF:21463
} else if((warmonly<0.8f)) {
// WRF:21464
pqcwd(1) = ((((((il5(1)*((((-qiacw(1))-qwfrz(1))-qwctfz(1))))-(il5(1)*(qiihr(1))))-qracw(1))-qrcnw(1))-qhacw(1))-qhlacw(1));
// WRF:21468
} else {
// WRF:21469
pqcwd(1) = ((-qracw(1))-qrcnw(1));
// WRF:21471
}
// WRF:21474
if(((pqcwd(1)<0.0f)&&(((-pqcwd(1))*dtp)>qx(1,lc)))) {
// WRF:21476
frac = lake_div((-lake_max(0.0f,qx(1,lc))),((pqcwd(1)*dtp)));
// WRF:21477
pqcwd(1) = ((-qx(1,lc))*dtpinv);
// WRF:21479
qiacw(1) = (frac*qiacw(1));
// WRF:21482
qwfrzc(1) = (frac*qwfrzc(1));
// WRF:21483
qwfrz(1) = (frac*qwfrz(1));
// WRF:21484
qwctfzc(1) = (frac*qwctfzc(1));
// WRF:21485
qwctfz(1) = (frac*qwctfz(1));
// WRF:21486
qracw(1) = (frac*qracw(1));
// WRF:21487
qsacw(1) = (frac*qsacw(1));
// WRF:21488
qhacw(1) = (frac*qhacw(1));
// WRF:21489
vhacw(1) = (frac*vhacw(1));
// WRF:21490
qrcnw(1) = (frac*qrcnw(1));
// WRF:21491
qwfrzp(1) = (frac*qwfrzp(1));
// WRF:21492
if((lhl>1)) {
// WRF:21493
qhlacw(1) = (frac*qhlacw(1));
// WRF:21494
vhlacw(1) = (frac*vhlacw(1));
// WRF:21495
}
// WRF:21499
}
// WRF:21502
}
// WRF:21506
if((warmonly<0.5f)) {
// WRF:21508
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21509
if((ffrzs<1.0f)) {
// WRF:21510
pqcii(1) = ((((((((il5(1)*qicicnt(1))+(il5(1)*(((((1.0f-cwfrz2snowfrac))*qwfrzc(1))+qwctfzc(1)))))+(il5(1)*(qicichr(1))))+qsmul(1))+qhmul1(1))+qhlmul1(1))+qsplinter(1))+qsplinter2(1));
// WRF:21518
}
// WRF:21520
pqcii(1) = (((pqcii(1)*((1.0f-ffrzs)))+(il5(1)*qidpv(1)))+(il5(1)*qiacw(1)));
// WRF:21524
pqcid(1) = ((((((il5(1)*(((((-qscni(1))-qscnvi(1))-qraci(1))-qsaci(1))))-qhaci(1))-qhlaci(1))+(il5(1)*qisbv(1)))+(((1.f-il5(1)))*qimlr(1)))-qhcni(1));
// WRF:21533
}
// WRF:21536
} else if((warmonly<0.8f)) {
// WRF:21538
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21539
pqcii(1) = ((((((((((il5(1)*qicicnt(1))*((1.f-ffrzs)))+((il5(1)*(((((1.0f-cwfrz2snowfrac))*qwfrzc(1))+qwctfzc(1))))*((1.f-ffrzs))))+((il5(1)*(qicichr(1)))*((1.f-ffrzs))))+qhmul1(1))+qhlmul1(1))+qsplinter(1))+qsplinter2(1))+(il5(1)*qidpv(1)))+(il5(1)*qiacw(1)));
// WRF:21555
pqcid(1) = (((+il5(1))*qisbv(1))+(((1.f-il5(1)))*qimlr(1)));
// WRF:21564
}
// WRF:21566
}
// WRF:21571
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21572
if((warmonly<0.5f)) {
// WRF:21573
pqrwi(1) = ((((qracw(1)+qrcnw(1))+lake_max(0.0f,qrcev(1)))+(((1-il5(1)))*(((((-qhmlr(1))-qsmlr(1))-qhlmlr(1))-qimlr(1)))))-qrshr(1));
// WRF:21584
pqrwd(1) = ((((((il5(1)*(((-qiacr(1))-qrfrz(1))))-qsacr(1))-qhacr(1))-qhlacr(1))-qwcnr(1))+lake_min(0.0f,qrcev(1)));
// WRF:21588
} else if((warmonly<0.8f)) {
// WRF:21589
pqrwi(1) = (((((qracw(1)+qrcnw(1))+lake_max(0.0f,qrcev(1)))+(((1-il5(1)))*(((-qhlmlr(1))-qhmlr(1)))))-qhshr(1))-qhlshr(1));
// WRF:21596
pqrwd(1) = ((((il5(1)*((-qrfrz(1))))-qhacr(1))-qhlacr(1))+lake_min(0.0f,qrcev(1)));
// WRF:21601
} else {
// WRF:21602
pqrwi(1) = ((qracw(1)+qrcnw(1))+lake_max(0.0f,qrcev(1)));
// WRF:21604
pqrwd(1) = lake_min(0.0f,qrcev(1));
// WRF:21605
}
// WRF:21609
if(((pqrwd(1)<0.0f)&&(((-((pqrwd(1)+pqrwi(1))))*dtp)>qx(1,lr)))) {
// WRF:21611
frac = lake_div((((-qx(1,lr))+(pqrwi(1)*dtp))),((pqrwd(1)*dtp)));
// WRF:21614
pqwvi(1) = ((pqwvi(1)+lake_min(0.0f,qrcev(1)))-(frac*lake_min(0.0f,qrcev(1))));
// WRF:21617
pqwvd(1) = ((pqwvd(1)+lake_max(0.0f,qrcev(1)))-(frac*lake_max(0.0f,qrcev(1))));
// WRF:21621
qiacr(1) = (frac*qiacr(1));
// WRF:21622
qiacrf(1) = (frac*qiacrf(1));
// WRF:21623
qiacrs(1) = (frac*qiacrs(1));
// WRF:21624
viacrf(1) = (frac*viacrf(1));
// WRF:21625
qrfrz(1) = (frac*qrfrz(1));
// WRF:21626
qrfrzs(1) = (frac*qrfrzs(1));
// WRF:21627
qrfrzf(1) = (frac*qrfrzf(1));
// WRF:21628
vrfrzf(1) = (frac*vrfrzf(1));
// WRF:21629
qsacr(1) = (frac*qsacr(1));
// WRF:21630
qhacr(1) = (frac*qhacr(1));
// WRF:21631
vhacr(1) = (frac*vhacr(1));
// WRF:21632
qrcev(1) = (frac*qrcev(1));
// WRF:21633
qhlacr(1) = (frac*qhlacr(1));
// WRF:21634
vhlacr(1) = (frac*vhlacr(1));
// WRF:21635
qhcev(1) = (frac*qhcev(1));
// WRF:21636
qhlcev(1) = (frac*qhlcev(1));
// WRF:21639
if((warmonly<0.5f)) {
// WRF:21640
pqrwd(1) = (((((il5(1)*((((-qiacr(1))-qrfrz(1))-qsacr(1))))-qhacr(1))-qhlacr(1))-qwcnr(1))+lake_min(0.0f,qrcev(1)));
// WRF:21644
} else if((warmonly<0.8f)) {
// WRF:21645
pqrwd(1) = ((((il5(1)*((-qrfrz(1))))-qhacr(1))-qhlacr(1))+lake_min(0.0f,qrcev(1)));
// WRF:21650
} else {
// WRF:21651
pqrwd(1) = lake_min(0.0f,qrcev(1));
// WRF:21652
}
// WRF:21657
if((qrcev(1)!=0.0f)) {
// WRF:21658
pqwvi(1) = ((((((((-lake_min(0.0f,qrcev(1)))-lake_min(0.0f,qhcev(1)))-lake_min(0.0f,qhlcev(1)))-lake_min(0.0f,qscev(1)))-qhsbv(1))-qhlsbv(1))-qssbv(1))-(il5(1)*qisbv(1)));
// WRF:21668
pqwvd(1) = ((((((-lake_max(0.0f,qrcev(1)))-lake_max(0.0f,qhcev(1)))-lake_max(0.0f,qhlcev(1)))-lake_max(0.0f,qscev(1)))+(il5(1)*(((((-qiint(1))-qhdpv(1))-qsdpv(1))-qhldpv(1)))))-(il5(1)*qidpv(1)));
// WRF:21677
}
// WRF:21681
}
// WRF:21684
}
// WRF:21686
if((warmonly<0.5f)) {
// WRF:21691
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21692
pqswi(1) = (((((((il5(1)*(((((((qscni(1)+qsaci(1))+qsdpv(1))+qscnvi(1))+(ifrzs*((qiacrs(1)+qrfrzs(1)))))+(il5(1)*((((((qwfrzc(1)+qwctfzc(1))+qicichr(1)))*ffrzs)+((((1.0f-ffrzs))*cwfrz2snowfrac)*qwfrz(1))))))+(il2(1)*qsacr(1)))))+((il5(1)*qicicnt(1))*ffrzs))+(il3(1)*((qiacrf(1)+qracif(1)))))+lake_max(0.0f,qscev(1)))+qsacw(1))+qscnh(1))+(ffrzs*(((((qsmul(1)+qhmul1(1))+qhlmul1(1))+qsplinter(1))+qsplinter2(1)))));
// WRF:21706
pqswd(1) = ((((((((((-qracs(1))*((1-il2(1))))-qhacs(1))-qhlacs(1))-qhcns(1))+(((1-il5(1)))*qsmlr(1)))+qsshr(1))+qssbv(1))+lake_min(0.0f,qscev(1)))-qsmul(1));
// WRF:21717
if(((imixedphase==0)&&(pqswd(1)<0.0f))) {
// WRF:21718
if(((qx(1,ls)+(dtp*((pqswi(1)+pqswd(1)))))<0.0f)) {
// WRF:21719
frac = lake_div((((-qx(1,ls))+(pqswi(1)*dtp))),((pqswd(1)*dtp)));
// WRF:21721
pqswd(1) = (frac*pqswd(1));
// WRF:21723
qracs(1) = (frac*qracs(1));
// WRF:21724
qhacs(1) = (frac*qhacs(1));
// WRF:21725
qhlacs(1) = (frac*qhlacs(1));
// WRF:21726
qhcns(1) = (frac*qhcns(1));
// WRF:21727
qsmlr(1) = (frac*qsmlr(1));
// WRF:21728
qsshr(1) = (frac*qsshr(1));
// WRF:21729
qssbv(1) = (frac*qssbv(1));
// WRF:21730
qsmul(1) = (frac*qsmul(1));
// WRF:21731
if((qscev(1)<0.0f)) {qscev(1) = (frac*qscev(1));}
// WRF:21733
}
// WRF:21734
}
// WRF:21736
pqcii(1) = ((pqcii(1)+(((1.f-ifrzs))*qrfrzs(1)))+(((1.f-ifrzs))*qiacrs(1)));
// WRF:21740
}
// WRF:21745
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21746
pqhwi(1) = ((((((((((((+il5(1))*((((ffrzh*ifrzg)*qrfrzf(1))+(((((1-il3(1)))*ffrzh)*ifiacrg)*((qiacrf(1)+qracif(1)))))))+(((1-il2(1)))*((qracs(1)+qsacr(1)))))+(il5(1)*(qhdpv(1))))+lake_max(0.0f,qhcev(1)))+qhacr(1))+qhacw(1))+qhacs(1))+qhaci(1))+(f2h*qhcns(1)))+(f2h*qhcni(1)))+qhcnhl(1));
// WRF:21754
pqhwd(1) = (((((((qhshr(1)+(((1-il5(1)))*qhmlr(1)))+qhsbv(1))+lake_min(0.0f,qhcev(1)))-qhmul1(1))-qhlcnh(1))-qscnh(1))-(ffrzh*((qsplinter(1)+qsplinter2(1)))));
// WRF:21764
}
// WRF:21770
if((lhl>1)) {
// WRF:21772
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21773
pqhli(1) = ((((((((+il5(1))*((qhldpv(1)+(((((1.0f-ifrzg))*qrfrzf(1))+(((1.0f-ifiacrg))*((qiacrf(1)+qracif(1)))))))))+lake_max(0.0f,qhlcev(1)))+qhlacr(1))+qhlacw(1))+qhlacs(1))+qhlaci(1))+qhlcnh(1));
// WRF:21779
pqhld(1) = (((((qhlshr(1)+(((1-il5(1)))*qhlmlr(1)))+qhlsbv(1))+lake_min(0.0f,qhlcev(1)))-qhlmul1(1))-qhcnhl(1));
// WRF:21787
if((imixedphase==0)) {
// WRF:21788
frac = 0.0f;
// WRF:21789
if(((qx(1,lhl)+(dtp*((pqhli(1)+pqhld(1)))))<0.0f)) {
// WRF:21792
frac = lake_div((((-qx(1,lhl))+(pqhli(1)*dtp))),((pqhld(1)*dtp)));
// WRF:21794
qhlmlr(1) = (frac*qhlmlr(1));
// WRF:21795
qhlsbv(1) = (frac*qhlsbv(1));
// WRF:21796
qhcnhl(1) = (frac*qhcnhl(1));
// WRF:21797
qhlmul1(1) = (frac*qhlmul1(1));
// WRF:21798
if((qhlcev(1)<0.0f)) {qhlcev(1) = (frac*qhlcev(1));}
// WRF:21800
pqhld(1) = (frac*pqhld(1));
// WRF:21802
}
// WRF:21803
}
// WRF:21806
}
// WRF:21808
}
// WRF:21810
} else if((warmonly<0.8f)) {
// WRF:21814
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21815
pqhwi(1) = ((((((+il5(1))*ifrzg)*(qrfrzf(1)))+(il5(1)*(qhdpv(1))))+qhacr(1))+qhacw(1));
// WRF:21819
pqhwd(1) = (((((qhshr(1)-qhlcnh(1))-qhmul1(1))-qsplinter(1))-qsplinter2(1))+(((1-il5(1)))*qhmlr(1)));
// WRF:21825
}
// WRF:21830
if((lhl>1)) {
// WRF:21832
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21833
pqhli(1) = ((((((+il5(1))*(qhldpv(1)))+((il5(1)*((1.0f-ifrzg)))*(qrfrzf(1))))+qhlacr(1))+qhlacw(1))+qhlcnh(1));
// WRF:21839
pqhld(1) = ((((qhlshr(1)+(((1-il5(1)))*qhlmlr(1)))+qhlsbv(1))-qhlmul1(1))-qhcnhl(1));
// WRF:21846
}
// WRF:21848
}
// WRF:21850
}
// WRF:21856
for(int _s0=1;_s0<=ngs;++_s0) {vhmlr(_s0) = 0.0f;}
// WRF:21857
for(int _s0=1;_s0<=ngs;++_s0) {vhlmlr(_s0) = 0.0f;}
// WRF:21858
for(int _s0=1;_s0<=ngs;++_s0) {vhfzh(_s0) = 0.0f;}
// WRF:21859
for(int _s0=1;_s0<=ngs;++_s0) {vhlfzhl(_s0) = 0.0f;}
// WRF:21861
if(mixedphase) {
// WRF:21862
} else {
// WRF:21865
for(int _s0=1;_s0<=ngs;++_s0) {vhmlr(_s0) = qhmlr((1+(_s0-1)));}
// WRF:21869
for(int _s0=1;_s0<=ngs;++_s0) {vhlmlr(_s0) = qhlmlr((1+(_s0-1)));}
// WRF:21873
}
// WRF:21880
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:21882
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21888
zhdsv(1) = 0.0f;
// WRF:21890
if((ffrzh>0.0f)) {
// WRF:21891
ziacr(1) = 0.0f;
// WRF:21892
ziacrf(1) = 0.0f;
// WRF:21893
}
// WRF:21895
zhcns(1) = 0.0f;
// WRF:21896
zhcni(1) = 0.0f;
// WRF:21897
zhacs(1) = 0.0f;
// WRF:21898
zhaci(1) = 0.0f;
// WRF:21900
}
// WRF:21902
if((lzh>1)) {
// WRF:21903
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:21906
if(((qx(1,lh)>coeff_qxmin{}(lh))&&(cx(1,lh)>0.0f))) {
// WRF:21907
tmp = lake_div(qx(1,lh),cx(1,lh));
// WRF:21908
alp = lake_max(alphamin,alpha(1,lh));
// WRF:21910
g1 = g1x(1,lh);
// WRF:21913
zhaci(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*(((2.f*(tmp))*qhaci(1))));
// WRF:21914
zhacs(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*(((2.f*(tmp))*qhacs(1))));
// WRF:21916
if(((!mixedphase)&&(ibinhmlr<1))) {
// WRF:21917
zhmlr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*((((2.f*tmp)*qhmlr(1))-(lake_pow(tmp,2)*chmlr(1)))));
// WRF:21918
}
// WRF:21920
zhshr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*((((2.f*tmp)*qhshr(1))-(lake_pow(tmp,2)*chshr(1)))));
// WRF:21923
if((((lzr>0)&&(qhshr(1)!=0.0f))&&(chshrr(1)!=0.0f))) {
// WRF:21935
if((temg(1)>=tfr)) {
// WRF:21940
if(((((shedalp+alpha(1,lh)))*xdia(1,lh,1))<sheddiam)) {
// WRF:21941
z1 = ((g1*lake_pow((lake_div((6.0f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhshr(1),2),chshrr(1))));
// WRF:21942
} else {
// WRF:21943
z1 = ((g1shr*lake_pow((lake_div((6.0f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhshr(1),2),chshrr(1))));
// WRF:21944
}
// WRF:21945
zhshrr(1) = z1;
// WRF:21948
} else {
// WRF:21949
zhshrr(1) = ((g1shr*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhshr(1),2),chshrr(1))));
// WRF:21950
}
// WRF:21952
zhshrr(1) = lake_min(0.0f,zhshrr(1));
// WRF:21953
}
// WRF:21955
if((zhshr(1)>0.0f)) {
// WRF:21956
; /* native diagnostic only */
// WRF:21957
; /* native diagnostic only */
// WRF:21958
; /* native diagnostic only */
// WRF:21959
; /* native diagnostic only */
// WRF:21961
asm("trap;"); return; /* native STOP */
// WRF:21962
}
// WRF:21967
qtmp = ((qhdpv(1)+qhcev(1))+qhsbv(1));
// WRF:21968
ctmp = ((chdpv(1)+chcev(1))+chsbv(1));
// WRF:21970
zhdsv(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*((((2.f*(tmp))*qtmp)-(lake_pow(tmp,2)*ctmp))));
// WRF:21972
alp = lake_max(alphahacx,alpha(1,lh));
// WRF:21974
g1 = g1x(1,lh);
// WRF:21976
if(true) {
// WRF:21977
if((qhacr(1)>0.0f)) {
// WRF:21982
zhacr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*(((2.f*(lake_div(qx(1,lh),cx(1,lh))))*qhacr(1))));
// WRF:21988
if((z>zx(1,lh))) {
// WRF:21990
} else {
// WRF:21992
}
// WRF:21993
}
// WRF:22001
if((qhacw(1)>0.0f)) {
// WRF:22003
zhacw(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*(((2.f*(lake_div(qx(1,lh),cx(1,lh))))*qhacw(1))));
// WRF:22006
if((z>zx(1,lh))) {
// WRF:22008
}
// WRF:22009
}
// WRF:22011
} else {
// WRF:22013
if(((qhacw(1)>0.0f)||(qhacr(1)>0.0f))) {
// WRF:22014
z = lake_div(((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*1000.f)))),2))*(lake_pow(((qx(1,lh)+(dtp*(((qhacr(1)+qhacw(1))-qhmul1(1)))))),2))),(cx(1,lh)));
// WRF:22016
if((z>zx(1,lh))) {
// WRF:22017
zhacw(1) = (((z-zx(1,lh)))*dtpinv);
// WRF:22018
}
// WRF:22019
}
// WRF:22021
}
// WRF:22023
if(((qhlcnh(1)>0.0f)&&(ihlcnh<2))) {
// WRF:22024
zhlcnh(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lh))))),2))*((((2.f*(tmp))*qhlcnh(1))-(lake_pow(tmp,2)*chlcnh(1)))));
// WRF:22025
}
// WRF:22026
}
// WRF:22028
if(((((ffrzh*qiacrf(1))>0.0f)&&(cx(1,lr)>0.0f))&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:22029
tmp = lake_div(qx(1,lr),cx(1,lr));
// WRF:22032
if((imurain==3)) {
// WRF:22034
ziacr(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*((alpha(1,lr)+2.f))),((lake_pow(coeff_xdn0{}(lr),2)*((alpha(1,lr)+1.f)))))*((((2.f*tmp)*qiacrf(1))-(lake_pow(tmp,2)*ciacrf(1)))));
// WRF:22036
} else {
// WRF:22037
ziacr(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*g1x(1,lr)),(lake_pow(coeff_xdn0{}(lr),2)))*((((2.f*tmp)*qiacrf(1))-(lake_pow(tmp,2)*ciacrf(1)))));
// WRF:22039
}
// WRF:22040
ziacr(1) = lake_min(ziacr(1),zxmxd(1,lr));
// WRF:22042
ziacrf(1) = (lake_pow((lake_div(xdn(1,lr),coeff_xdnmx{}(lh))),2)*ziacr(1));
// WRF:22045
}
// WRF:22049
if((((ffrzh*qrfrzf(1))>0.0f)&&(cx(1,lr)>0.0f))) {
// WRF:22050
tmp = lake_div(qx(1,lr),cx(1,lr));
// WRF:22053
if((imurain==3)) {
// WRF:22054
zrfrz(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*((alpha(1,lr)+2.f))),((lake_pow(coeff_xdn0{}(lr),2)*((alpha(1,lr)+1.f)))))*((((2.f*tmp)*qrfrzf(1))-(lake_pow(tmp,2)*crfrzf(1)))));
// WRF:22056
zrfrzf(1) = (lake_pow((lake_div(xdn(1,lr),xdn(1,lh))),2)*zrfrz(1));
// WRF:22057
} else if(((imurain==1)&&(ibiggopt!=2))) {
// WRF:22060
zrfrz(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*g1x(1,lr)),(lake_pow(coeff_xdn0{}(lr),2)))*((((2.f*tmp)*qrfrz(1))-(lake_pow(tmp,2)*crfrz(1)))));
// WRF:22062
zrfrzf(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*g1x(1,lr)),(lake_pow(rhofrz,2)))*((((2.f*tmp)*qrfrzf(1))-(lake_pow(tmp,2)*crfrzf(1)))));
// WRF:22064
}
// WRF:22065
zrfrz(1) = lake_min(zrfrz(1),((lake_max(0.4f,lake_div(qrfrz(1),qx(1,lr)))*zx(1,lr))*dtpinv));
// WRF:22071
}
// WRF:22073
if(((lhl>1)&&(qhcnhl(1)>0.0f))) {
// WRF:22074
tmp = lake_div(qx(1,lhl),cx(1,lhl));
// WRF:22075
zhcnhl(1) = ((g1x(1,lhl)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*((((2.f*(tmp))*qhcnhl(1))-(lake_pow(tmp,2)*chcnhl(1)))));
// WRF:22077
}
// WRF:22079
if(((((qhcns(1)>0.0f)&&(chcns(1)>0.0f))&&(cx(1,ls)>cxmin))&&(vhcns(1)>0))) {
// WRF:22080
tmp = lake_div(qx(1,ls),cx(1,ls));
// WRF:22081
r = lake_div((rho0(1)*qhcns(1)),vhcns(1));
// WRF:22082
if((imusnow==3)) {
// WRF:22083
zhcns(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*((alpha(1,ls)+2.f))),((lake_pow(r,2)*((alpha(1,ls)+1.f)))))*((((2.f*tmp)*qhcns(1))-(lake_pow(tmp,2)*chcns(1)))));
// WRF:22085
} else {
// WRF:22086
; /* native diagnostic only */
// WRF:22087
asm("trap;"); return; /* native STOP */
// WRF:22088
}
// WRF:22089
}
// WRF:22091
if(((((qhcni(1)>0.0f)&&(chcni(1)>0.0f))&&(cx(1,li)>cxmin))&&(vhcni(1)>0))) {
// WRF:22092
tmp = lake_div(qx(1,li),cx(1,li));
// WRF:22093
r = lake_div((rho0(1)*qhcni(1)),vhcni(1));
// WRF:22094
zhcni(1) = (lake_div(((3.6476f*lake_pow(rho0(1),2))*((alpha(1,li)+2.f))),((lake_pow(r,2)*((alpha(1,li)+1.f)))))*((((2.f*tmp)*qhcni(1))-(lake_pow(tmp,2)*chcni(1)))));
// WRF:22096
}
// WRF:22099
pzhwi(1) = (((((((((((+ifrzg)*ffrzh)*((zrfrzf(1)+((il5(1)*ifiacrg)*(ziacrf(1))))))+zhacw(1))+zhacr(1))+zhcnhl(1))+zhacs(1))+zhaci(1))+(f2h*zhcni(1)))+(f2h*zhcns(1)))+lake_max(0.0f,zhdsv(1)));
// WRF:22111
pzhwd(1) = ((((0.0f+(((1-il5(1)))*zhmlr(1)))+zhshr(1))+lake_min(0.0f,zhdsv(1)))-(il5(1)*zhlcnh(1)));
// WRF:22118
if((((1==44)&&(1==23))||((dtp*((pqhwi(1)+pqhwd(1))))>coeff_qxmin{}(lh)))) {
// WRF:22123
}
// WRF:22133
}
// WRF:22135
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:22137
}
// WRF:22143
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22145
zhldsv(1) = 0.0f;
// WRF:22146
zhlacr(1) = 0.0f;
// WRF:22147
zhlacw(1) = 0.0f;
// WRF:22149
}
// WRF:22151
if(((lzhl>1)||(((lzr>1)&&(lnhl>1))))) {
// WRF:22153
if(((ndebug>0)&&(my_rank>=0))) {; /* native diagnostic only */}
// WRF:22155
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22157
if(((qx(1,lhl)>coeff_qxmin{}(lhl))&&(cx(1,lhl)>0.0f))) {
// WRF:22158
tmp = lake_div(qx(1,lhl),cx(1,lhl));
// WRF:22159
alp = lake_max(alphamin,alpha(1,lhl));
// WRF:22161
g1 = g1x(1,lhl);
// WRF:22163
if(((((!mixedphase)&&(qhlmlr(1)!=0.0f))&&(chlmlr(1)!=0.0f))&&(ibinhlmlr<1))) {
// WRF:22164
zhlmlr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*((((2.f*tmp)*qhlmlr(1))-(lake_pow(tmp,2)*chlmlr(1)))));
// WRF:22165
}
// WRF:22167
zhlshr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*((((2.f*tmp)*qhlshr(1))-(lake_pow(tmp,2)*chlshr(1)))));
// WRF:22168
if((((lzr>1)&&(qhlshr(1)!=0.0f))&&(chlshrr(1)!=0.0f))) {
// WRF:22169
if((temg(1)>=tfr)) {
// WRF:22174
if(((((shedalp+alpha(1,lhl)))*xdia(1,lhl,1))<sheddiam)) {
// WRF:22175
z1 = ((g1*lake_pow((lake_div((6.0f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhlshr(1),2),chlshrr(1))));
// WRF:22176
} else {
// WRF:22177
z1 = ((g1shr*lake_pow((lake_div((6.0f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhlshr(1),2),chlshrr(1))));
// WRF:22178
}
// WRF:22179
zhlshrr(1) = z1;
// WRF:22182
} else {
// WRF:22183
zhlshrr(1) = ((g1shr*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhlshr(1),2),chlshrr(1))));
// WRF:22184
}
// WRF:22186
zhlshrr(1) = lake_min(0.0f,zhlshrr(1));
// WRF:22187
}
// WRF:22189
if((zhlshr(1)>0.0f)) {
// WRF:22190
; /* native diagnostic only */
// WRF:22191
; /* native diagnostic only */
// WRF:22192
; /* native diagnostic only */
// WRF:22193
; /* native diagnostic only */
// WRF:22195
asm("trap;"); return; /* native STOP */
// WRF:22196
}
// WRF:22201
qtmp = (qhldpv(1)+qhlcev(1));
// WRF:22202
ctmp = (chldpv(1)+chlcev(1));
// WRF:22204
zhldsv(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*((((2.f*(tmp))*qtmp)-(lake_pow(tmp,2)*ctmp))));
// WRF:22206
alp = lake_max(alphahacx,alpha(1,lhl));
// WRF:22208
g1 = g1x(1,lhl);
// WRF:22210
if(true) {
// WRF:22211
if((qhlacr(1)>0.0f)) {
// WRF:22213
zhlacr(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*(((2.f*(tmp))*qhlacr(1))));
// WRF:22221
}
// WRF:22226
if((qhlacw(1)>0.0f)) {
// WRF:22227
alp = lake_max(3.0f,(alpha(1,lhl)+1.f));
// WRF:22228
g1 = lake_div(((((6.0f+alp))*((5.0f+alp)))*((4.0f+alp))),(((((3.0f+alp))*((2.0f+alp)))*((1.0f+alp)))));
// WRF:22232
zhlacw(1) = ((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lhl))))),2))*(((2.f*tmp)*qhlacw(1))));
// WRF:22237
g1 = g1x(1,lhl);
// WRF:22238
}
// WRF:22240
} else {
// WRF:22242
if(((qhlacw(1)>0.0f)||(qhlacr(1)>0.0f))) {
// WRF:22243
z = lake_div(((g1*lake_pow((lake_div((6.f*rho0(1)),((pi*1000.f)))),2))*(lake_pow(((qx(1,lhl)+(dtp*(((qhlacr(1)+qhlacw(1))-qhlmul1(1)))))),2))),(cx(1,lhl)));
// WRF:22245
if((z>zx(1,lhl))) {
// WRF:22246
zhlacw(1) = (((z-zx(1,lhl)))*dtpinv);
// WRF:22247
}
// WRF:22248
}
// WRF:22250
}
// WRF:22252
}
// WRF:22255
if((lzhl>1)) {
// WRF:22256
pzhli(1) = (((((ffrzh*((((((1.0f-ifrzg))*zrfrzf(1))+((il5(1)*((1.0f-ifiacrg)))*ziacrf(1))))))+(il5(1)*zhlcnh(1)))+zhlacw(1))+zhlacr(1))+lake_max(0.0f,zhldsv(1)));
// WRF:22264
pzhld(1) = ((((0.0f+(((1-il5(1)))*zhlmlr(1)))+zhlshr(1))-zhcnhl(1))+lake_min(0.0f,zhldsv(1)));
// WRF:22271
if((!((((-1.0f)<pzhli(1))&&(pzhli(1)<1.e20f))))) {
// WRF:22272
; /* native diagnostic only */
// WRF:22273
; /* native diagnostic only */
// WRF:22274
}
// WRF:22276
if((!((((-1.0e20f)<pzhld(1))&&(pzhld(1)<1.f))))) {
// WRF:22277
; /* native diagnostic only */
// WRF:22278
; /* native diagnostic only */
// WRF:22279
}
// WRF:22281
}
// WRF:22283
}
// WRF:22285
}
// WRF:22290
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:22292
if((lzr>1)) {
// WRF:22294
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22296
zracw(1) = 0.0f;
// WRF:22297
zracr(1) = 0.0f;
// WRF:22298
zrcev(1) = 0.0f;
// WRF:22299
zrach(1) = 0.0f;
// WRF:22300
zrachl(1) = 0.0f;
// WRF:22301
zsshr(1) = 0.0f;
// WRF:22302
zsshrr(1) = 0.0f;
// WRF:22304
zsmlrr(1) = 0.0f;
// WRF:22306
if(((qx(1,ls)>coeff_qxmin{}(ls))&&(((((csmlr(1)!=0.0f)||(csshr(1)!=0.0f))||(csmlrr(1)!=0.0f))||(csshrr(1)!=0.0f))))) {
// WRF:22308
tmp = lake_div(qx(1,ls),cx(1,ls));
// WRF:22309
g1 = lake_div((36.f*((coeff_xnu{}(ls)+2.0f))),((((coeff_xnu{}(ls)+1.0f))*lake_pow(pi,2))));
// WRF:22310
if((!mixedphase)) {
// WRF:22314
if((csmlrr(1)!=0.0f)) {
// WRF:22315
z1 = ((g1smlr*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qsmlr(1),2),csmlrr(1))));
// WRF:22316
zsmlrr(1) = z1;
// WRF:22317
}
// WRF:22318
}
// WRF:22323
if((csshrr(1)!=0.0f)) {
// WRF:22324
z1 = ((g1smlr*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qsshr(1),2),csshrr(1))));
// WRF:22325
zsshrr(1) = z1;
// WRF:22326
}
// WRF:22328
}
// WRF:22330
if((!mixedphase)) {
// WRF:22331
if((((zhmlr(1)<0.0f)&&(chmlrr(1)!=0.0f))&&(ibinhmlr==0))) {
// WRF:22332
tmp = lake_div(qx(1,lh),cx(1,lh));
// WRF:22339
if(((((shedalp+alpha(1,lh)))*xdia(1,lh,1))<sheddiam)) {
// WRF:22340
z1 = ((g1x(1,lh)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhmlr(1),2),chmlrr(1))));
// WRF:22341
} else {
// WRF:22342
z1 = ((lake_min(g1x(1,lh),g1shr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhmlr(1),2),chmlrr(1))));
// WRF:22343
}
// WRF:22344
zhmlrr(1) = z1;
// WRF:22347
}
// WRF:22352
if((((lhl>1)&&(qhlmlr(1)!=0))&&(ibinhlmlr==0))) {
// WRF:22353
tmp = lake_div(qx(1,lhl),cx(1,lhl));
// WRF:22361
if(((((shedalp+alpha(1,lhl)))*xdia(1,lhl,1))<sheddiam)) {
// WRF:22362
z1 = ((g1x(1,lhl)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhlmlr(1),2),chlmlrr(1))));
// WRF:22363
} else {
// WRF:22364
z1 = ((lake_min(g1x(1,lhl),g1shr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*(lake_div(lake_pow(qhlmlr(1),2),chlmlrr(1))));
// WRF:22366
}
// WRF:22367
zhlmlrr(1) = z1;
// WRF:22373
}
// WRF:22375
}
// WRF:22377
if(((qx(1,lr)>coeff_qxmin{}(lr))&&(cx(1,lr)>0.0f))) {
// WRF:22379
tmp = lake_div(qx(1,lr),cx(1,lr));
// WRF:22380
g1 = g1x(1,lr);
// WRF:22383
if(((qracw(1)>0.0f)&&(cx(1,lr)>0.0f))) {
// WRF:22384
zracw(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*1000.f)))),2))*(((2.f*tmp)*qracw(1))));
// WRF:22385
}
// WRF:22387
if(((cracr(1)>0.0f)&&(cx(1,lr)>0.0f))) {
// WRF:22388
zracr(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*1000.f)))),2))*((lake_pow(tmp,2)*cracr(1))));
// WRF:22389
}
// WRF:22391
qtmp = qrcev(1);
// WRF:22392
ctmp = crcev(1);
// WRF:22397
zrcev(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*((((2.f*(tmp))*qtmp)-(lake_pow(tmp,2)*ctmp))));
// WRF:22400
if((iferwisventr==2)) {
// WRF:22401
vent1 = lake_min(0.0f,(((((lake_div(12.f,((pii*xdn(1,lr)))))*lake_pow(xdia(1,lr,1),3))*fvce(1))*rwcap(1))*rwventz(1)));
// WRF:22402
zrcev(1) = lake_max(zrcev(1),vent1);
// WRF:22403
}
// WRF:22410
zrcev(1) = lake_max(zrcev(1),(-zxmxd(1,lr)));
// WRF:22412
if((qhacr(1)>0.0f)) {
// WRF:22413
zrach(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*((((2.f*(lake_div(qx(1,lr),cx(1,lr))))*qhacr(1))-(lake_pow(tmp,2)*chacr(1)))));
// WRF:22415
zrach(1) = lake_min(zrach(1),zxmxd(1,lr));
// WRF:22417
}
// WRF:22419
if(((lhl>1)&&(qhlacr(1)>0.0f))) {
// WRF:22420
zrachl(1) = ((g1x(1,lr)*lake_pow((lake_div((6.f*rho0(1)),((pi*xdn(1,lr))))),2))*((((2.f*(lake_div(qx(1,lr),cx(1,lr))))*qhlacr(1))-(lake_pow(tmp,2)*chlacr(1)))));
// WRF:22422
zrachl(1) = lake_min(zrachl(1),zxmxd(1,lr));
// WRF:22423
}
// WRF:22427
}
// WRF:22429
pzrwi(1) = (((((((((zrcnw(1)+zracw(1))+zracr(1))+lake_max(0.f,zrcev(1)))-(((1-il5(1)))*zsmlrr(1)))-zsshrr(1))-(((1-il5(1)))*zhmlrr(1)))-zhshrr(1))-(((1-il5(1)))*zhlmlrr(1)))-zhlshrr(1));
// WRF:22439
pzrwd(1) = (((((0.0f+lake_min(0.f,zrcev(1)))-zrach(1))-zrachl(1))-zrfrz(1))-(il5(1)*(ziacr(1))));
// WRF:22447
if((((zx(1,lr)+(dtp*((pzrwi(1)+pzrwd(1)))))<=0.0f)&&(qx(1,lr)>coeff_qxmin{}(lr)))) {
// WRF:22449
pzrwd(1) = (((-zx(1,lr))*dtpinv)-pzrwi(1));
// WRF:22450
}
// WRF:22452
}
// WRF:22454
}
// WRF:22461
if((coeff_lvol{}(ls)>1)) {
// WRF:22462
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22465
pvswi(1) = ((rho0(1)*((lake_div(((+il5(1))*((((((qscni(1)+qsaci(1))+qsdpv(1))+qscnvi(1))+(((1.f-ifrzs))*qiacrs(1)))+(((1.f-ifrzs))*qrfrzs(1))))),coeff_xdn0{}(ls))+lake_div((qsacr(1)),rimdn(1,ls)))))+vsacw(1));
// WRF:22474
pvswd(1) = (lake_div((rho0(1)*(pqswd(1))),coeff_xdn0{}(ls))-lake_div((rho0(1)*qsmul(1)),coeff_xdn0{}(ls)));
// WRF:22483
}
// WRF:22490
}
// WRF:22494
if((coeff_lvol{}(lh)>1)) {
// WRF:22495
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22501
pvhwi(1) = ((((((((rho0(1)*((lake_div(((+il5(1))*(((ifiacrg*ffrzh)*qracif(1)))),rhofrz)+((lake_div((il5(1)*qhdpv(1)),qhdpvdn)+lake_div(((qhacs(1)+qhaci(1))),qhacidn))))))+lake_div((rho0(1)*lake_max(0.0f,qhcev(1))),1000.f))+(f2h*vhcns(1)))+vhacr(1))+vhacw(1))+vhfzh(1))+(f2h*vhcni(1)))+((((ifiacrg*viacrf(1))+(ifrzg*vrfrzf(1))))*ffrzh));
// WRF:22516
pvhwd(1) = (((((rho0(1)*(lake_div((+(((((((1-il5(1)))*vhmlr(1))+qhsbv(1))+lake_min(0.0f,qhcev(1)))-qhmul1(1)))),xdn(1,lh))))-vhlcnh(1))+vhshdr(1))-vhsoak(1))-vscnh(1));
// WRF:22531
if(((lzh>1)&&(qx(1,lh)>coeff_qxmin{}(lh)))) {
// WRF:22534
xdn_new = lake_div((rho0(1)*((qx(1,lh)+(dtp*((pqhwi(1)+pqhwd(1))))))),((vx(1,lh)+(dtp*((pvhwi(1)+pvhwd(1)))))));
// WRF:22537
if(mixedphase) {
// WRF:22538
if((qxw(1,lh)>0.0f)) {
// WRF:22539
dnmx = coeff_xdnmx{}(lr);
// WRF:22540
} else {
// WRF:22541
dnmx = coeff_xdnmx{}(lh);
// WRF:22542
}
// WRF:22543
} else {
// WRF:22544
dnmx = coeff_xdnmx{}(lh);
// WRF:22545
}
// WRF:22547
xdn_new = lake_max(lake_min(xdn_new,dnmx),coeff_xdnmn{}(lh));
// WRF:22549
drhodt = (((xdn_new-xdn(1,lh)))*dtpinv);
// WRF:22551
zhwdn(1) = (lake_div((((-2.f)*g1x(1,lh))*lake_pow(((((rho0(1)*qx(1,lh))*6.f)*pii)),2)),((cx(1,lh)*lake_pow(xdn(1,lh),3))))*drhodt);
// WRF:22553
pzhwi(1) = (pzhwi(1)+lake_max(0.0f,zhwdn(1)));
// WRF:22554
pzhwd(1) = (pzhwd(1)+lake_min(0.0f,zhwdn(1)));
// WRF:22557
}
// WRF:22558
if((((false&&(ny==2))&&(1==9))&&(1==19))) {
// WRF:22560
; /* native diagnostic only */
// WRF:22561
; /* native diagnostic only */
// WRF:22563
; /* native diagnostic only */
// WRF:22564
; /* native diagnostic only */
// WRF:22565
; /* native diagnostic only */
// WRF:22566
; /* native diagnostic only */
// WRF:22567
; /* native diagnostic only */
// WRF:22568
; /* native diagnostic only */
// WRF:22569
; /* native diagnostic only */
// WRF:22570
; /* native diagnostic only */
// WRF:22571
; /* native diagnostic only */
// WRF:22572
; /* native diagnostic only */
// WRF:22573
; /* native diagnostic only */
// WRF:22574
; /* native diagnostic only */
// WRF:22575
; /* native diagnostic only */
// WRF:22576
; /* native diagnostic only */
// WRF:22577
; /* native diagnostic only */
// WRF:22578
; /* native diagnostic only */
// WRF:22579
; /* native diagnostic only */
// WRF:22580
; /* native diagnostic only */
// WRF:22581
; /* native diagnostic only */
// WRF:22582
; /* native diagnostic only */
// WRF:22583
; /* native diagnostic only */
// WRF:22584
; /* native diagnostic only */
// WRF:22585
; /* native diagnostic only */
// WRF:22586
; /* native diagnostic only */
// WRF:22587
; /* native diagnostic only */
// WRF:22588
; /* native diagnostic only */
// WRF:22589
; /* native diagnostic only */
// WRF:22590
; /* native diagnostic only */
// WRF:22591
; /* native diagnostic only */
// WRF:22592
; /* native diagnostic only */
// WRF:22593
; /* native diagnostic only */
// WRF:22594
; /* native diagnostic only */
// WRF:22598
; /* native diagnostic only */
// WRF:22599
; /* native diagnostic only */
// WRF:22600
; /* native diagnostic only */
// WRF:22601
; /* native diagnostic only */
// WRF:22602
; /* native diagnostic only */
// WRF:22605
}
// WRF:22608
}
// WRF:22610
}
// WRF:22618
if((lhl>1)) {
// WRF:22619
if((coeff_lvol{}(lhl)>1)) {
// WRF:22620
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22622
pvhli(1) = (((((((rho0(1)*(lake_div((+((((il5(1)*((lake_div((((((1.0f-ifiacrg))*ffrzh)*qracif(1))),rhofrz)+qhldpv(1))))+qhlacs(1))+qhlaci(1)))),500.f)))+lake_div((rho0(1)*lake_max(0.0f,qhlcev(1))),1000.f))+vhlcnhl(1))+((((((1.0f-ifiacrg))*ffrzh)*viacrf(1))+((((1.0f-ifrzg))*ffrzh)*vrfrzf(1)))))+vhlacr(1))+vhlacw(1))+vhlfzhl(1));
// WRF:22632
pvhld(1) = ((((rho0(1)*(lake_div((+(((qhlsbv(1)+lake_min(0.0f,qhlcev(1)))-qhlmul1(1)))),xdn(1,lhl))))+lake_div(((rho0(1)*((1-il5(1))))*vhlmlr(1)),xdn(1,lhl)))+vhlshdr(1))-vhlsoak(1));
// WRF:22640
if(((lzhl>1)&&(qx(1,lhl)>coeff_qxmin{}(lhl)))) {
// WRF:22643
xdn_new = lake_div((rho0(1)*((qx(1,lhl)+(dtp*((pqhli(1)+pqhld(1))))))),((vx(1,lhl)+(dtp*((pvhli(1)+pvhld(1)))))));
// WRF:22646
if(mixedphase) {
// WRF:22647
if((qxw(1,lhl)>0.0f)) {
// WRF:22648
dnmx = coeff_xdnmx{}(lr);
// WRF:22649
} else {
// WRF:22650
dnmx = coeff_xdnmx{}(lhl);
// WRF:22651
}
// WRF:22652
} else {
// WRF:22653
dnmx = coeff_xdnmx{}(lhl);
// WRF:22654
}
// WRF:22655
xdn_new = lake_max(lake_min(xdn_new,dnmx),coeff_xdnmn{}(lhl));
// WRF:22657
drhodt = (((xdn_new-xdn(1,lhl)))*dtpinv);
// WRF:22659
zhldn(1) = (lake_div((((-2.f)*g1x(1,lhl))*lake_pow(((((rho0(1)*qx(1,lhl))*6.f)*pii)),2)),((cx(1,lhl)*lake_pow(xdn(1,lhl),3))))*drhodt);
// WRF:22661
pzhli(1) = (pzhli(1)+lake_max(0.0f,zhldn(1)));
// WRF:22662
pzhld(1) = (pzhld(1)+lake_min(0.0f,zhldn(1)));
// WRF:22665
}
// WRF:22667
}
// WRF:22669
}
// WRF:22670
}
// WRF:22673
if((ndebug>=1)) {
// WRF:22674
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22676
ptotal(1) = 0.f;
// WRF:22677
ptotal(1) = ((((((((((((((ptotal(1)+pqwvi(1))+pqwvd(1))+pqcwi(1))+pqcwd(1))+pqcii(1))+pqcid(1))+pqrwi(1))+pqrwd(1))+pqswi(1))+pqswd(1))+pqhwi(1))+pqhwd(1))+pqhli(1))+pqhld(1));
// WRF:22689
}
// WRF:22691
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22693
if((((((ndebug>=0))&&(lake_abs(ptotal(1))>eqtot)))||(!(((ptotal(1)<1.0f)&&(ptotal(1)>(-1.0f))))))) {
// WRF:22702
; /* native diagnostic only */
// WRF:22705
; /* native diagnostic only */
// WRF:22706
; /* native diagnostic only */
// WRF:22707
; /* native diagnostic only */
// WRF:22708
; /* native diagnostic only */
// WRF:22709
; /* native diagnostic only */
// WRF:22710
; /* native diagnostic only */
// WRF:22711
; /* native diagnostic only */
// WRF:22712
; /* native diagnostic only */
// WRF:22713
if((lhl>1)) {; /* native diagnostic only */}
// WRF:22716
; /* native diagnostic only */
// WRF:22720
; /* native diagnostic only */
// WRF:22721
; /* native diagnostic only */
// WRF:22723
; /* native diagnostic only */
// WRF:22724
; /* native diagnostic only */
// WRF:22725
; /* native diagnostic only */
// WRF:22726
; /* native diagnostic only */
// WRF:22727
; /* native diagnostic only */
// WRF:22728
; /* native diagnostic only */
// WRF:22729
; /* native diagnostic only */
// WRF:22730
tmp = (((((((((((((pqwvi(1)+pqwvd(1))+pqcwi(1))+pqcwd(1))+pqcii(1))+pqcid(1))+pqrwi(1))+pqrwd(1))+pqswi(1))+pqswd(1))+pqhwi(1))+pqhwd(1))+pqhli(1))+pqhld(1));
// WRF:22738
; /* native diagnostic only */
// WRF:22739
; /* native diagnostic only */
// WRF:22744
; /* native diagnostic only */
// WRF:22745
; /* native diagnostic only */
// WRF:22747
; /* native diagnostic only */
// WRF:22748
; /* native diagnostic only */
// WRF:22749
; /* native diagnostic only */
// WRF:22750
; /* native diagnostic only */
// WRF:22751
; /* native diagnostic only */
// WRF:22752
; /* native diagnostic only */
// WRF:22753
; /* native diagnostic only */
// WRF:22754
; /* native diagnostic only */
// WRF:22755
; /* native diagnostic only */
// WRF:22756
; /* native diagnostic only */
// WRF:22757
; /* native diagnostic only */
// WRF:22758
; /* native diagnostic only */
// WRF:22759
; /* native diagnostic only */
// WRF:22760
; /* native diagnostic only */
// WRF:22761
; /* native diagnostic only */
// WRF:22762
; /* native diagnostic only */
// WRF:22764
; /* native diagnostic only */
// WRF:22765
; /* native diagnostic only */
// WRF:22767
; /* native diagnostic only */
// WRF:22768
; /* native diagnostic only */
// WRF:22769
; /* native diagnostic only */
// WRF:22770
; /* native diagnostic only */
// WRF:22771
; /* native diagnostic only */
// WRF:22772
; /* native diagnostic only */
// WRF:22773
; /* native diagnostic only */
// WRF:22774
; /* native diagnostic only */
// WRF:22775
; /* native diagnostic only */
// WRF:22776
; /* native diagnostic only */
// WRF:22777
; /* native diagnostic only */
// WRF:22778
; /* native diagnostic only */
// WRF:22779
; /* native diagnostic only */
// WRF:22780
; /* native diagnostic only */
// WRF:22781
; /* native diagnostic only */
// WRF:22782
; /* native diagnostic only */
// WRF:22783
; /* native diagnostic only */
// WRF:22784
; /* native diagnostic only */
// WRF:22785
; /* native diagnostic only */
// WRF:22786
; /* native diagnostic only */
// WRF:22787
; /* native diagnostic only */
// WRF:22788
; /* native diagnostic only */
// WRF:22789
; /* native diagnostic only */
// WRF:22790
; /* native diagnostic only */
// WRF:22791
; /* native diagnostic only */
// WRF:22792
; /* native diagnostic only */
// WRF:22793
; /* native diagnostic only */
// WRF:22798
; /* native diagnostic only */
// WRF:22799
; /* native diagnostic only */
// WRF:22801
; /* native diagnostic only */
// WRF:22802
; /* native diagnostic only */
// WRF:22803
; /* native diagnostic only */
// WRF:22804
; /* native diagnostic only */
// WRF:22807
; /* native diagnostic only */
// WRF:22808
; /* native diagnostic only */
// WRF:22809
; /* native diagnostic only */
// WRF:22810
; /* native diagnostic only */
// WRF:22811
; /* native diagnostic only */
// WRF:22812
; /* native diagnostic only */
// WRF:22813
; /* native diagnostic only */
// WRF:22814
; /* native diagnostic only */
// WRF:22815
; /* native diagnostic only */
// WRF:22818
; /* native diagnostic only */
// WRF:22819
; /* native diagnostic only */
// WRF:22820
; /* native diagnostic only */
// WRF:22821
; /* native diagnostic only */
// WRF:22822
; /* native diagnostic only */
// WRF:22823
; /* native diagnostic only */
// WRF:22824
; /* native diagnostic only */
// WRF:22825
; /* native diagnostic only */
// WRF:22826
; /* native diagnostic only */
// WRF:22827
; /* native diagnostic only */
// WRF:22828
; /* native diagnostic only */
// WRF:22829
; /* native diagnostic only */
// WRF:22831
; /* native diagnostic only */
// WRF:22832
; /* native diagnostic only */
// WRF:22834
; /* native diagnostic only */
// WRF:22835
; /* native diagnostic only */
// WRF:22836
; /* native diagnostic only */
// WRF:22837
; /* native diagnostic only */
// WRF:22838
; /* native diagnostic only */
// WRF:22839
; /* native diagnostic only */
// WRF:22840
; /* native diagnostic only */
// WRF:22841
; /* native diagnostic only */
// WRF:22842
; /* native diagnostic only */
// WRF:22843
; /* native diagnostic only */
// WRF:22844
; /* native diagnostic only */
// WRF:22845
; /* native diagnostic only */
// WRF:22846
; /* native diagnostic only */
// WRF:22847
; /* native diagnostic only */
// WRF:22848
; /* native diagnostic only */
// WRF:22849
; /* native diagnostic only */
// WRF:22850
; /* native diagnostic only */
// WRF:22851
; /* native diagnostic only */
// WRF:22852
; /* native diagnostic only */
// WRF:22853
; /* native diagnostic only */
// WRF:22856
; /* native diagnostic only */
// WRF:22857
; /* native diagnostic only */
// WRF:22858
; /* native diagnostic only */
// WRF:22859
; /* native diagnostic only */
// WRF:22860
; /* native diagnostic only */
// WRF:22861
; /* native diagnostic only */
// WRF:22862
; /* native diagnostic only */
// WRF:22863
; /* native diagnostic only */
// WRF:22864
; /* native diagnostic only */
// WRF:22865
; /* native diagnostic only */
// WRF:22866
; /* native diagnostic only */
// WRF:22867
; /* native diagnostic only */
// WRF:22868
; /* native diagnostic only */
// WRF:22872
; /* native diagnostic only */
// WRF:22873
; /* native diagnostic only */
// WRF:22875
; /* native diagnostic only */
// WRF:22876
; /* native diagnostic only */
// WRF:22877
; /* native diagnostic only */
// WRF:22878
; /* native diagnostic only */
// WRF:22879
; /* native diagnostic only */
// WRF:22880
; /* native diagnostic only */
// WRF:22881
; /* native diagnostic only */
// WRF:22882
; /* native diagnostic only */
// WRF:22883
; /* native diagnostic only */
// WRF:22884
; /* native diagnostic only */
// WRF:22885
; /* native diagnostic only */
// WRF:22886
; /* native diagnostic only */
// WRF:22887
; /* native diagnostic only */
// WRF:22888
; /* native diagnostic only */
// WRF:22890
; /* native diagnostic only */
// WRF:22891
; /* native diagnostic only */
// WRF:22892
; /* native diagnostic only */
// WRF:22893
; /* native diagnostic only */
// WRF:22894
; /* native diagnostic only */
// WRF:22895
; /* native diagnostic only */
// WRF:22896
; /* native diagnostic only */
// WRF:22897
; /* native diagnostic only */
// WRF:22900
; /* native diagnostic only */
// WRF:22901
; /* native diagnostic only */
// WRF:22903
; /* native diagnostic only */
// WRF:22904
; /* native diagnostic only */
// WRF:22905
; /* native diagnostic only */
// WRF:22906
; /* native diagnostic only */
// WRF:22907
; /* native diagnostic only */
// WRF:22908
; /* native diagnostic only */
// WRF:22909
; /* native diagnostic only */
// WRF:22910
; /* native diagnostic only */
// WRF:22911
; /* native diagnostic only */
// WRF:22912
; /* native diagnostic only */
// WRF:22913
; /* native diagnostic only */
// WRF:22914
; /* native diagnostic only */
// WRF:22915
; /* native diagnostic only */
// WRF:22916
; /* native diagnostic only */
// WRF:22917
; /* native diagnostic only */
// WRF:22918
; /* native diagnostic only */
// WRF:22919
; /* native diagnostic only */
// WRF:22920
; /* native diagnostic only */
// WRF:22921
; /* native diagnostic only */
// WRF:22922
; /* native diagnostic only */
// WRF:22923
; /* native diagnostic only */
// WRF:22924
; /* native diagnostic only */
// WRF:22925
; /* native diagnostic only */
// WRF:22928
; /* native diagnostic only */
// WRF:22929
; /* native diagnostic only */
// WRF:22931
; /* native diagnostic only */
// WRF:22932
; /* native diagnostic only */
// WRF:22933
; /* native diagnostic only */
// WRF:22934
; /* native diagnostic only */
// WRF:22935
; /* native diagnostic only */
// WRF:22936
; /* native diagnostic only */
// WRF:22937
; /* native diagnostic only */
// WRF:22938
; /* native diagnostic only */
// WRF:22939
; /* native diagnostic only */
// WRF:22940
; /* native diagnostic only */
// WRF:22941
; /* native diagnostic only */
// WRF:22942
; /* native diagnostic only */
// WRF:22943
; /* native diagnostic only */
// WRF:22944
; /* native diagnostic only */
// WRF:22945
; /* native diagnostic only */
// WRF:22950
; /* native diagnostic only */
// WRF:22951
; /* native diagnostic only */
// WRF:22953
}
// WRF:22955
}
// WRF:22958
}
// WRF:22963
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:22964
if((warmonly<0.5f)) {
// WRF:22965
pfrz(1) = (((((1-il5(1)))*(((qhmlr(1)+qsmlr(1))+qhlmlr(1))))+((il5(1)*((1-imixedphase)))*(((((((((((qsacw(1)+qhacw(1))+qhlacw(1))+qsacr(1))+qhacr(1))+qhlacr(1))+qsshr(1))+qhshr(1))+qhlshr(1))+qrfrz(1))+qiacr(1)))))+(il5(1)*((((qwfrz(1)+qwctfz(1))+qiihr(1))+qiacw(1)))));
// WRF:22980
pmlt(1) = (((1-il5(1)))*(((qhmlr(1)+qsmlr(1))+qhlmlr(1))));
// WRF:22985
psub(1) = (((((il5(1)*((((((+qsdpv(1))+qhdpv(1))+qhldpv(1))+qidpv(1))+qisbv(1))))+qssbv(1))+qhsbv(1))+qhlsbv(1))+(il5(1)*(qiint(1))));
// WRF:22993
pvap(1) = ((((qrcev(1)+qhcev(1))+qscev(1))+qhlcev(1))+qfcev(1));
// WRF:22995
pevap(1) = ((((lake_min(0.0f,qrcev(1))+lake_min(0.0f,qhcev(1)))+lake_min(0.0f,qscev(1)))+lake_min(0.0f,qhlcev(1)))+lake_min(0.0f,qfcev(1)));
// WRF:22999
pdep(1) = ((il5(1)*(((((+qsdpv(1))+qhdpv(1))+qhldpv(1))+qidpv(1))))+(il5(1)*(qiint(1))));
// WRF:23005
} else if((warmonly<0.8f)) {
// WRF:23006
pfrz(1) = (((((1-il5(1)))*((qhmlr(1)+qhlmlr(1))))+(il5(1)*((qhfzh(1)+qhlfzhl(1)))))+(il5(1)*((((((((((((+qhshr(1))+qhlshr(1))+qrfrz(1))+qwfrz(1))+qwctfz(1))+qiihr(1))+qiacw(1))+qhacw(1))+qhlacw(1))+qhacr(1))+qhlacr(1)))));
// WRF:23018
psub(1) = ((0.0f+(il5(1)*(((((+qhdpv(1))+qhldpv(1))+qidpv(1))+qisbv(1)))))+(il5(1)*(qiint(1))));
// WRF:23024
pvap(1) = (((qrcev(1)+qhcev(1))+qhlcev(1))+qfcev(1));
// WRF:23026
} else {
// WRF:23027
pfrz(1) = 0.0f;
// WRF:23028
psub(1) = 0.0f;
// WRF:23029
pvap(1) = qrcev(1);
// WRF:23030
}
// WRF:23031
ptem(1) = ((lake_div(1.f,pi0(1)))*((((felfcp(1)*pfrz(1))+(felscp(1)*psub(1)))+(felvcp(1)*pvap(1)))));
// WRF:23036
thetap(1) = (thetap(1)+(dtp*ptem(1)));
// WRF:23037
ptem2(1) = ptem(1);
// WRF:23038
if((eqtset>2)) {
// WRF:23039
pipert(1) = (pipert(1)+(((((felfpi(1)*pfrz(1))+(felspi(1)*psub(1)))+(felvpi(1)*pvap(1))))*dtp));
// WRF:23042
}
// WRF:23043
}
// WRF:23052
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23055
qwvp(1) = (qwvp(1)+(dtp*((pqwvi(1)+pqwvd(1)))));
// WRF:23057
qx(1,lc) = (qx(1,lc)+(dtp*((pqcwi(1)+pqcwd(1)))));
// WRF:23059
qx(1,lr) = (qx(1,lr)+(dtp*((pqrwi(1)+pqrwd(1)))));
// WRF:23061
qx(1,li) = (qx(1,li)+(dtp*((pqcii(1)+pqcid(1)))));
// WRF:23063
qx(1,ls) = (qx(1,ls)+(dtp*((pqswi(1)+pqswd(1)))));
// WRF:23065
qx(1,lh) = (qx(1,lh)+(dtp*((pqhwi(1)+pqhwd(1)))));
// WRF:23068
if((lhl>1)) {
// WRF:23069
qx(1,lhl) = (qx(1,lhl)+(dtp*((pqhli(1)+pqhld(1)))));
// WRF:23071
}
// WRF:23074
}
// WRF:23078
if(ldovol) {
// WRF:23080
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23082
if((coeff_lvol{}(ls)>1)) {
// WRF:23083
vx(1,ls) = (vx(1,ls)+(dtp*((pvswi(1)+pvswd(1)))));
// WRF:23085
}
// WRF:23087
if((coeff_lvol{}(lh)>1)) {
// WRF:23088
vx(1,lh) = (vx(1,lh)+(dtp*((pvhwi(1)+pvhwd(1)))));
// WRF:23091
}
// WRF:23093
if((lhl>1)) {
// WRF:23094
if((coeff_lvol{}(lhl)>1)) {
// WRF:23095
vx(1,lhl) = (vx(1,lhl)+(dtp*((pvhli(1)+pvhld(1)))));
// WRF:23098
}
// WRF:23099
}
// WRF:23101
}
// WRF:23103
}
// WRF:23110
if((ipconc>=1)) {
// WRF:23111
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23112
cx(1,li) = (cx(1,li)+(dtp*((pccii(1)+pccid(1)))));
// WRF:23114
cina(1) = (cina(1)+(pccin(1)*dtp));
// WRF:23115
if((ipconc>=2)) {
// WRF:23116
cx(1,lc) = (cx(1,lc)+(dtp*((pccwi(1)+pccwd(1)))));
// WRF:23118
}
// WRF:23119
if((ipconc>=3)) {
// WRF:23120
cx(1,lr) = (cx(1,lr)+(dtp*((pcrwi(1)+pcrwd(1)))));
// WRF:23122
}
// WRF:23123
if((ipconc>=4)) {
// WRF:23124
cx(1,ls) = (cx(1,ls)+(dtp*((pcswi(1)+pcswd(1)))));
// WRF:23126
}
// WRF:23127
if((ipconc>=5)) {
// WRF:23128
cx(1,lh) = (cx(1,lh)+(dtp*((pchwi(1)+pchwd(1)))));
// WRF:23130
if((lhl>1)) {
// WRF:23131
cx(1,lhl) = (cx(1,lhl)+(dtp*((pchli(1)+pchld(1)))));
// WRF:23137
}
// WRF:23138
}
// WRF:23139
if((ipconc>=6)) {
// WRF:23140
if((lzr>1)) {
// WRF:23141
zx(1,lr) = (zx(1,lr)+(dtp*((pzrwi(1)+pzrwd(1)))));
// WRF:23143
}
// WRF:23144
if((lzs>1)) {
// WRF:23145
zx(1,ls) = (zx(1,ls)+(dtp*((pzswi(1)+pzswd(1)))));
// WRF:23147
}
// WRF:23148
if((lzh>1)) {
// WRF:23149
zx(1,lh) = (zx(1,lh)+(dtp*((pzhwi(1)+pzhwd(1)))));
// WRF:23151
}
// WRF:23152
if((lzhl>1)) {
// WRF:23153
zx(1,lhl) = (zx(1,lhl)+(dtp*((pzhli(1)+pzhld(1)))));
// WRF:23158
}
// WRF:23159
}
// WRF:23160
}
// WRF:23161
}
// WRF:23163
if(has_wetscav) {
// WRF:23164
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23165
evapprod2d(1,1) = (-((((qrcev(1)+qssbv(1))+qhsbv(1))+qhlsbv(1))));
// WRF:23166
rainprod2d(1,1) = (((((((((qrcnw(1)+qracw(1))+qsacw(1))+qhacw(1))+qhlacw(1))+qraci(1))+qsaci(1))+qhaci(1))+qhlaci(1))+qscni(1));
// WRF:23168
}
// WRF:23169
}
// WRF:23175
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:23186
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23187
pqs(1) = lake_div((380.0f),(pres(1)));
// WRF:23188
theta(1) = (thetap(1)+theta0(1));
// WRF:23189
qvap(1) = lake_max(((qwvp(1)+qv0(1))),0.0f);
// WRF:23190
temg(1) = (theta(1)*pk(1));
// WRF:23191
}
// WRF:23195
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23196
qcwtmp(1) = qx(1,lc);
// WRF:23197
ptimlw(1) = 0.0f;
// WRF:23198
}
// WRF:23200
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23201
qitmp(1) = qx(1,li);
// WRF:23202
if(((temg(1)>tfr)&&(qitmp(1)>0.0f))) {
// WRF:23204
qx(1,lc) = (qx(1,lc)+qitmp(1));
// WRF:23206
ptem(1) = (ptem(1)+(((lake_div(1.f,pi0(1)))*felfcp(1))*(((-qitmp(1))*dtpinv))));
// WRF:23209
if((eqtset>2)) {
// WRF:23210
pipert(1) = (pipert(1)-((felfpi(1)*qitmp(1))));
// WRF:23211
}
// WRF:23212
pmlt(1) = (pmlt(1)-(qitmp(1)*dtpinv));
// WRF:23213
scx(1,lc) = (scx(1,lc)+scx(1,li));
// WRF:23214
thetap(1) = (thetap(1)-(fcc3(1)*qitmp(1)));
// WRF:23216
ptimlw(1) = (((-fcc3(1))*qitmp(1))*dtpinv);
// WRF:23217
cx(1,lc) = (cx(1,lc)+cx(1,li));
// WRF:23218
qx(1,li) = 0.0f;
// WRF:23219
cx(1,li) = 0.0f;
// WRF:23220
scx(1,li) = 0.0f;
// WRF:23221
vx(1,li) = 0.0f;
// WRF:23222
qitmp(1) = 0.0f;
// WRF:23223
}
// WRF:23224
}
// WRF:23236
if((warmonly<0.8f)) {
// WRF:23238
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23239
qcwtmp(1) = qx(1,lc);
// WRF:23240
ptwfzi(1) = 0.0f;
// WRF:23241
}
// WRF:23243
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23249
ctmp = 0.0f;
// WRF:23250
frac = 0.0f;
// WRF:23251
qtmp = 0.0f;
// WRF:23258
if(((((((temg(1)<(thnuc+0.f)))||(((temg(1)<(thnuc+2.f))&&(ibfc>=3)))))&&(qx(1,lc)>0.0f))&&((((ipconc<2)||(ibfc==0))||(ibfc==2))))) {
// WRF:23261
if((ibfc>=3)) {
// WRF:23262
frac = lake_max(0.25f,lake_min(1.f,lake_div(((((thnuc+2.f))-temg(1))),4.0f)));
// WRF:23263
} else if(((ibfc!=2)||(ipconc<2))) {
// WRF:23264
frac = lake_max(0.25f,lake_min(1.f,lake_div(((((thnuc+1.f))-temg(1))),4.0f)));
// WRF:23265
} else {
// WRF:23266
volt = (gfk_exp((16.2f+(1.0f*temcg(1))))*1.0e-6f);
// WRF:23270
cwfrz(1) = (cx(1,lc)*gfk_exp(lake_div((-volt),xv(1,lc))));
// WRF:23272
qtmp = (((cwfrz(1)*coeff_xdn0{}(lc))*rhoinv(1))*((volt+xv(1,lc))));
// WRF:23273
frac = lake_div(qtmp,qx(1,lc));
// WRF:23277
}
// WRF:23278
qtmp = (frac*qx(1,lc));
// WRF:23280
if(((ibfc==4)&&(lis>=1))) {
// WRF:23281
qx(1,lis) = (qx(1,lis)+qtmp);
// WRF:23282
} else {
// WRF:23283
qx(1,li) = (qx(1,li)+qtmp);
// WRF:23284
}
// WRF:23285
pfrz(1) = (pfrz(1)+(qtmp*dtpinv));
// WRF:23286
ptem(1) = (ptem(1)+(((lake_div(1.f,pi0(1)))*felfcp(1))*((qtmp*dtpinv))));
// WRF:23290
if((eqtset>2)) {
// WRF:23291
pipert(1) = (pipert(1)+(felfpi(1)*qtmp));
// WRF:23292
}
// WRF:23295
if((coeff_lvol{}(li)>1)) {vx(1,li) = (vx(1,li)+lake_div((rho0(1)*qtmp),coeff_xdn0{}(li)));}
// WRF:23297
if((ipconc>=2)) {
// WRF:23298
ctmp = (frac*cx(1,lc));
// WRF:23300
if(((ibfc==4)&&(lis>=1))) {
// WRF:23301
cx(1,lis) = (cx(1,lis)+ctmp);
// WRF:23302
} else {
// WRF:23303
cx(1,li) = (cx(1,li)+ctmp);
// WRF:23304
}
// WRF:23305
} else {
// WRF:23306
ctmp = 0.0f;
// WRF:23307
if((below_qc>qx(1,lc))) {
// WRF:23308
qtmp = (frac*below_qc);
// WRF:23311
ctmp = lake_div(((cx(1,lc)*qx(1,lc))*rho0(1)),qtmp);
// WRF:23312
} else {
// WRF:23313
cx(1,lc) = lake_div(((lake_max(0.0f,wvel(1))*dtp)*cwccn),gz(1,jgs,1));
// WRF:23315
cx(1,lc) = cwccn;
// WRF:23316
}
// WRF:23318
if((ipconc>=1)) {cx(1,li) = lake_min(ccimx,(cx(1,li)+cx(1,lc)));}
// WRF:23319
}
// WRF:23321
sctmp = (frac*scx(1,lc));
// WRF:23323
scx(1,li) = (scx(1,li)+sctmp);
// WRF:23329
thetap(1) = (thetap(1)+(fcc3(1)*qtmp));
// WRF:23330
ptwfzi(1) = ((fcc3(1)*qtmp)*dtpinv);
// WRF:23331
qx(1,lc) = (qx(1,lc)-qtmp);
// WRF:23332
cx(1,lc) = (cx(1,lc)-ctmp);
// WRF:23333
scx(1,lc) = (scx(1,lc)-sctmp);
// WRF:23334
}
// WRF:23335
}
// WRF:23337
}
// WRF:23345
for(int _s0=1;_s0<=ngs;++_s0) {qcond(_s0) = 0.0f;}
// WRF:23347
if(((ipconc<=1)&&lwsm6)) {
// WRF:23348
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23350
qcwtmp(1) = qx(1,lc);
// WRF:23351
theta(1) = (thetap(1)+theta0(1));
// WRF:23352
temgtmp = temg(1);
// WRF:23356
temg(1) = (theta(1)*pk(1));
// WRF:23357
temcg(1) = (temg(1)-tfr);
// WRF:23358
ltemq = (lake_div(((temg(1)-163.15f)),fqsat)+1.5f);
// WRF:23359
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:23361
qvs(1) = (pqs(1)*tabqvs(ltemq));
// WRF:23363
if(((((qvap(1)>qvs(1))||(qx(1,lc)>coeff_qxmin{}(lc))))&&(temg(1)>tfrh))) {
// WRF:23364
tmp = lake_div(((qvap(1)-qvs(1))),((1.f+lake_div((qvs(1)*lake_pow(felv(1),2)),(((cp*rw)*lake_pow(temg(1),2)))))));
// WRF:23365
qcond(1) = lake_min(lake_max(0.0f,tmp),((qvap(1)-qvs(1))));
// WRF:23366
if(((qx(1,lc)>coeff_qxmin{}(lc))&&(tmp<0.0f))) {
// WRF:23367
qcond(1) = lake_max(tmp,(-qx(1,lc)));
// WRF:23368
}
// WRF:23369
qwvp(1) = (qwvp(1)-qcond(1));
// WRF:23370
qvap(1) = (qvap(1)-qcond(1));
// WRF:23371
qx(1,lc) = lake_max(0.0f,(qx(1,lc)+qcond(1)));
// WRF:23372
thetap(1) = (thetap(1)+lake_div((felvcp(1)*qcond(1)),(pi0(1))));
// WRF:23374
}
// WRF:23376
}
// WRF:23378
}
// WRF:23381
if(((ipconc<=1)&&(!lwsm6))) {
// WRF:23384
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23385
qx(1,lv) = lake_max(0.0f,qvap(1));
// WRF:23386
qx(1,lc) = lake_max(0.0f,qx(1,lc));
// WRF:23387
qx(1,li) = lake_max(0.0f,qx(1,li));
// WRF:23388
qitmp(1) = qx(1,li);
// WRF:23389
}
// WRF:23392
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23393
qcwtmp(1) = qx(1,lc);
// WRF:23394
qitmp(1) = qx(1,li);
// WRF:23395
theta(1) = (thetap(1)+theta0(1));
// WRF:23396
temgtmp = temg(1);
// WRF:23397
temg(1) = (theta(1)*((pinit(1)+p2(1,jgs,1))));
// WRF:23398
temsav = temg(1);
// WRF:23399
thsave(1) = thetap(1);
// WRF:23400
temcg(1) = (temg(1)-tfr);
// WRF:23401
tqvcon = (temg(1)-cbw);
// WRF:23402
ltemq = (lake_div(((temg(1)-163.15f)),fqsat)+1.5f);
// WRF:23403
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:23405
qvs(1) = (pqs(1)*tabqvs(ltemq));
// WRF:23406
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:23407
qss(1) = qvs(1);
// WRF:23408
if((temg(1)<tfr)) {
// WRF:23409
if(((qx(1,lc)>=0.0f)&&(qitmp(1)<=coeff_qxmin{}(li)))) {qss(1) = qvs(1);}
// WRF:23411
if(((qx(1,lc)<=coeff_qxmin{}(lc))&&(qitmp(1)>coeff_qxmin{}(li)))) {qss(1) = qis(1);}
// WRF:23413
if(((qx(1,lc)>coeff_qxmin{}(lc))&&(qitmp(1)>coeff_qxmin{}(li)))) {qss(1) = lake_div((((qx(1,lc)*qvs(1))+(qitmp(1)*qis(1)))),((qx(1,lc)+qitmp(1))));}
// WRF:23416
}
// WRF:23417
}
// WRF:23421
for(itertd=1;itertd<=2;itertd+=1) {
// WRF:23423
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23427
qitmp(1) = qx(1,li);
// WRF:23428
fcci(1) = 0.0f;
// WRF:23429
fcip(1) = 0.0f;
// WRF:23430
dqcw(1) = 0.0f;
// WRF:23431
dqci(1) = 0.0f;
// WRF:23432
dqwv(1) = ((qx(1,lv)-qss(1)));
// WRF:23436
if((dqwv(1)<0.f)) {
// WRF:23437
if((qx(1,lc)>(-dqwv(1)))) {
// WRF:23438
dqcw(1) = dqwv(1);
// WRF:23439
dqwv(1) = 0.f;
// WRF:23440
} else {
// WRF:23441
dqcw(1) = (-qx(1,lc));
// WRF:23442
dqwv(1) = (dqwv(1)+qx(1,lc));
// WRF:23443
}
// WRF:23445
if((qitmp(1)>(-dqwv(1)))) {
// WRF:23446
dqci(1) = dqwv(1);
// WRF:23447
dqwv(1) = 0.f;
// WRF:23448
} else {
// WRF:23449
dqci(1) = (-qitmp(1));
// WRF:23450
dqwv(1) = (dqwv(1)+qitmp(1));
// WRF:23451
}
// WRF:23453
qwvp(1) = (qwvp(1)-((dqcw(1)+dqci(1))));
// WRF:23458
qitmp(1) = qx(1,li);
// WRF:23459
if((qitmp(1)>=coeff_qxmin{}(li))) {
// WRF:23460
fcci(1) = lake_div(qx(1,li),(qitmp(1)));
// WRF:23461
} else {
// WRF:23462
fcci(1) = 1.0f;
// WRF:23463
}
// WRF:23464
qx(1,lc) = (qx(1,lc)+dqcw(1));
// WRF:23465
qx(1,li) = (qx(1,li)+(dqci(1)*fcci(1)));
// WRF:23466
thetap(1) = (thetap(1)+(lake_div(1.f,pi0(1))*(((felvcp(1)*dqcw(1))+(felscp(1)*dqci(1))))));
// WRF:23470
if((eqtset>2)) {
// WRF:23471
pipert(1) = (pipert(1)+((((felspi(1)*dqci(1))+(felvpi(1)*dqcw(1))))*dtp));
// WRF:23474
}
// WRF:23476
}
// WRF:23480
if((dqwv(1)>=0.f)) {
// WRF:23484
qitmp(1) = qx(1,li);
// WRF:23485
fracl(1) = 1.0f;
// WRF:23486
fraci(1) = 0.0f;
// WRF:23487
if(((temg(1)<tfr)&&(temg(1)>thnuc))) {
// WRF:23488
fracl(1) = lake_max(lake_min(1.f,lake_div(((temg(1)-233.15f)),(20.f))),0.0f);
// WRF:23489
fraci(1) = (1.0f-fracl(1));
// WRF:23490
}
// WRF:23491
if((temg(1)<=thnuc)) {
// WRF:23492
fraci(1) = 1.0f;
// WRF:23493
fracl(1) = 0.0f;
// WRF:23494
}
// WRF:23495
fraci(1) = (1.0f-fracl(1));
// WRF:23497
gamss = lake_div((((felvcp(1)*fracl(1))+(felscp(1)*fraci(1)))),(pi0(1)));
// WRF:23500
if((temg(1)<tfr)) {
// WRF:23501
if(((qx(1,lc)>=0.0f)&&(qitmp(1)<=coeff_qxmin{}(li)))) {
// WRF:23502
dqvcnd(1) = lake_div(dqwv(1),((1.f+lake_div((fcqv1(1)*qss(1)),(lake_pow(((temg(1)-cbw)),2))))));
// WRF:23504
}
// WRF:23505
if(((qx(1,lc)==0.0f)&&(qitmp(1)>coeff_qxmin{}(li)))) {
// WRF:23506
dqvcnd(1) = lake_div(dqwv(1),((1.f+lake_div((fcqv2(1)*qss(1)),(lake_pow(((temg(1)-cbi)),2))))));
// WRF:23508
}
// WRF:23509
if(((qx(1,lc)>0.0f)&&(qitmp(1)>coeff_qxmin{}(li)))) {
// WRF:23510
cdw = lake_div(((caw*pi0(1))*tfrcbw),(lake_pow(((temg(1)-cbw)),2)));
// WRF:23511
cdi = lake_div(((cai*pi0(1))*tfrcbi),(lake_pow(((temg(1)-cbi)),2)));
// WRF:23512
denom1 = (qx(1,lc)+qitmp(1));
// WRF:23513
denom2 = (1.0f+lake_div((gamss*((((qx(1,lc)*qvs(1))*cdw)+((qitmp(1)*qis(1))*cdi)))),denom1));
// WRF:23515
dqvcnd(1) = lake_div(dqwv(1),denom2);
// WRF:23516
}
// WRF:23518
}
// WRF:23520
if((temg(1)>=tfr)) {
// WRF:23521
dqvcnd(1) = lake_div(dqwv(1),((1.f+lake_div((fcqv1(1)*qss(1)),(lake_pow(((temg(1)-cbw)),2))))));
// WRF:23523
}
// WRF:23525
delqci1 = qx(1,li);
// WRF:23527
if((qitmp(1)>coeff_qxmin{}(li))) {
// WRF:23528
fcci(1) = lake_div(qx(1,li),(qitmp(1)));
// WRF:23529
} else {
// WRF:23530
fcci(1) = 1.0f;
// WRF:23531
}
// WRF:23533
dqcw(1) = (dqvcnd(1)*fracl(1));
// WRF:23534
dqci(1) = (dqvcnd(1)*fraci(1));
// WRF:23536
thetap(1) = (thetap(1)+lake_div((((felvcp(1)*dqcw(1))+(felscp(1)*dqci(1)))),(pi0(1))));
// WRF:23540
if((eqtset>2)) {
// WRF:23541
pipert(1) = (pipert(1)+((((0+(felspi(1)*dqci(1)))+(felvpi(1)*dqcw(1))))*dtp));
// WRF:23544
}
// WRF:23546
qwvp(1) = (qwvp(1)-(dqvcnd(1)));
// WRF:23547
qx(1,lc) = (qx(1,lc)+dqcw(1));
// WRF:23549
qx(1,li) = (qx(1,li)+(dqci(1)*fcci(1)));
// WRF:23550
qitmp(1) = qx(1,li);
// WRF:23555
}
// WRF:23556
}
// WRF:23558
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23559
qitmp(1) = qx(1,li);
// WRF:23560
theta(1) = (thetap(1)+theta0(1));
// WRF:23561
temg(1) = (theta(1)*pk(1));
// WRF:23562
qvap(1) = lake_max(((qwvp(1)+qv0(1))),0.0f);
// WRF:23563
temcg(1) = (temg(1)-tfr);
// WRF:23564
tqvcon = (temg(1)-cbw);
// WRF:23565
ltemq = (lake_div(((temg(1)-163.15f)),fqsat)+1.5f);
// WRF:23566
ltemq = lake_min(nqsat,lake_max(1,ltemq));
// WRF:23567
qvs(1) = (pqs(1)*tabqvs(ltemq));
// WRF:23568
qis(1) = (pqs(1)*tabqis(ltemq));
// WRF:23569
qx(1,lc) = lake_max(0.0f,qx(1,lc));
// WRF:23570
qitmp(1) = lake_max(0.0f,qitmp(1));
// WRF:23571
qx(1,lv) = lake_max(0.0f,qvap(1));
// WRF:23585
qss(1) = qvs(1);
// WRF:23586
if((temg(1)<tfr)) {
// WRF:23587
if(((qx(1,lc)>=0.0f)&&(qitmp(1)<=coeff_qxmin{}(li)))) {qss(1) = qvs(1);}
// WRF:23589
if(((qx(1,lc)<=coeff_qxmin{}(lc))&&(qitmp(1)>coeff_qxmin{}(li)))) {qss(1) = qis(1);}
// WRF:23591
if(((qx(1,lc)>coeff_qxmin{}(lc))&&(qitmp(1)>coeff_qxmin{}(li)))) {qss(1) = lake_div((((qx(1,lc)*qvs(1))+(qitmp(1)*qis(1)))),((qx(1,lc)+qitmp(1))));}
// WRF:23594
}
// WRF:23597
}
// WRF:23601
}
// WRF:23603
}
// WRF:23612
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:23620
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23621
t0(1,jy,1) = temg(1);
// WRF:23622
}
// WRF:23645
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:23647
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23649
an(1,jy,1,lt) = (theta0(1)+thetap(1));
// WRF:23651
an(1,jy,1,lv) = (qwvp(1)+qv0(1));
// WRF:23653
if((eqtset>2)) {
// WRF:23654
p2(1,jy,1) = pipert(1);
// WRF:23655
}
// WRF:23658
#pragma unroll
for(il=lc;il<=lhab;il+=1) {
// WRF:23659
if((coeff_ido{}(il)==1)) {
// WRF:23660
if(((lf>1)&&(il==lf))) {
// WRF:23661
lfsave(1,1) = an(1,jy,1,il);
// WRF:23662
lfsave(1,2) = qx(1,il);
// WRF:23663
}
// WRF:23664
an(1,jy,1,il) = (qx(1,il)+lake_min(an(1,jy,1,il),0.0f));
// WRF:23666
qx(1,il) = an(1,jy,1,il);
// WRF:23667
}
// WRF:23668
}
// WRF:23670
if((lcina>1)) {
// WRF:23671
an(1,jy,1,lcina) = cina(1);
// WRF:23672
}
// WRF:23682
if((ipconc>=6)) {
// WRF:23683
#pragma unroll
for(il=lr;il<=lhab;il+=1) {
// WRF:23684
if((coeff_lz{}(il)>1)) {
// WRF:23685
if(((lf>1)&&(il==lf))) {
// WRF:23686
lfsave(1,3) = an(1,jy,1,coeff_lz{}(il));
// WRF:23687
lfsave(1,4) = zx(1,il);
// WRF:23688
}
// WRF:23690
an(1,jy,1,coeff_lz{}(il)) = (zx(1,il)+lake_min(an(1,jy,1,coeff_lz{}(il)),0.0f));
// WRF:23692
zx(1,il) = an(1,jy,1,coeff_lz{}(il));
// WRF:23694
}
// WRF:23695
}
// WRF:23697
}
// WRF:23699
}
// WRF:23702
if((ipconc>=1)) {
// WRF:23703
#pragma unroll
for(il=lc;il<=lhab;il+=1) {
// WRF:23707
if(((ipconc>=coeff_ipc{}(il))&&(coeff_ido{}(il)>0))) {
// WRF:23709
if(((ipconc>=4)&&(coeff_ipc{}(il)>=1))) {
// WRF:23714
if(((coeff_lz{}(il)<=1)||(ioldlimiter==1))) {
// WRF:23717
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23718
if((qx(1,il)<=0.0f)) {
// WRF:23719
cx(1,il) = 0.0f;
// WRF:23720
} else {
// WRF:23721
if((cx(1,il)>cxmin)) {
// WRF:23724
xv(1,il) = lake_div((rho0(1)*qx(1,il)),((xdn(1,il)*cx(1,il))));
// WRF:23731
if((((((imaxdiaopt==1)||(il==lc))||(il==li))||(((il==lr)&&(imurain==3))))||(((il==ls)&&(imusnow==3))))) {
// WRF:23733
xvbarmax = coeff_xvmx{}(il);
// WRF:23734
} else if((imaxdiaopt==2)) {
// WRF:23735
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((3.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:23736
} else if((imaxdiaopt==3)) {
// WRF:23737
xvbarmax = lake_div(coeff_xvmx{}(il),(lake_div(lake_pow(((4.f+alpha(1,il))),3),(((((3.f+alpha(1,il)))*((2.f+alpha(1,il))))*((1.f+alpha(1,il))))))));
// WRF:23738
} else {
// WRF:23739
xvbarmax = coeff_xvmx{}(il);
// WRF:23740
}
// WRF:23742
tmp = 1.0f;
// WRF:23743
if((il==ls)) {
// WRF:23744
xvbarmax = (xvbarmax*lake_max(1.f,lake_div(100.f,lake_min(100.f,xdn(1,ls)))));
// WRF:23745
}
// WRF:23747
if(((xv(1,il)<coeff_xvmn{}(il))||(xv(1,il)>xvbarmax))) {
// WRF:23748
xv(1,il) = lake_min(xvbarmax,xv(1,il));
// WRF:23749
xv(1,il) = lake_max(coeff_xvmn{}(il),xv(1,il));
// WRF:23750
cx(1,il) = lake_div((rho0(1)*qx(1,il)),((xv(1,il)*xdn(1,il))));
// WRF:23751
}
// WRF:23753
}
// WRF:23759
}
// WRF:23760
}
// WRF:23762
} else {
// WRF:23763
if(((il==lr)&&(imurain==3))) {
// WRF:23768
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23771
if(((iresetmoments==1)||(iresetmoments==il))) {
// WRF:23772
if((zx(1,lr)<=zxmin)) {
// WRF:23773
qx(1,lv) = (qx(1,lv)+qx(1,il));
// WRF:23774
qx(1,lr) = 0.0f;
// WRF:23775
cx(1,lr) = 0.0f;
// WRF:23776
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,lr));
// WRF:23777
an(1,jgs,1,lr) = qx(1,lr);
// WRF:23778
an(1,jgs,1,coeff_ln{}(lr)) = cx(1,lr);
// WRF:23779
} else if((cx(1,lr)<=cxmin)) {
// WRF:23780
qx(1,lv) = (qx(1,lv)+qx(1,il));
// WRF:23781
zx(1,lr) = 0.0f;
// WRF:23782
qx(1,lr) = 0.0f;
// WRF:23783
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,lr));
// WRF:23784
an(1,jgs,1,lr) = qx(1,lr);
// WRF:23785
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:23786
}
// WRF:23787
}
// WRF:23789
if((qx(1,lr)>coeff_qxmin{}(lr))) {
// WRF:23791
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((xdn(1,lr)*lake_max(1.0e-11f,cx(1,lr)))));
// WRF:23792
if((xv(1,lr)>coeff_xvmx{}(lr))) {
// WRF:23795
} else if((xv(1,lr)<coeff_xvmn{}(lr))) {
// WRF:23796
xv(1,lr) = coeff_xvmn{}(lr);
// WRF:23797
cx(1,lr) = lake_div((rho0(1)*qx(1,lr)),((coeff_xvmn{}(lr)*xdn(1,lr))));
// WRF:23798
}
// WRF:23800
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:23802
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23803
z = zx(1,il);
// WRF:23804
qr = qx(1,il);
// WRF:23805
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),((z*lake_pow(xdn(1,lr),2))));
// WRF:23807
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>0.0f))) {
// WRF:23809
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23810
chw = cx(1,il);
// WRF:23811
qr = qx(1,il);
// WRF:23812
zx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),((lake_pow(xdn(1,lr),2)*chw)));
// WRF:23813
an(1,jgs,1,coeff_lz{}(lr)) = zx(1,lr);
// WRF:23815
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:23819
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:23820
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23822
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23823
z = zx(1,il);
// WRF:23824
qr = qx(1,il);
// WRF:23825
cx(1,il) = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),(((z*1000.f)*1000)));
// WRF:23826
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:23827
}
// WRF:23829
if((zx(1,lr)>0.0f)) {
// WRF:23830
xv(1,lr) = lake_div((rho0(1)*qx(1,lr)),((1000.f*cx(1,lr))));
// WRF:23831
vr = xv(1,lr);
// WRF:23832
qr = qx(1,lr);
// WRF:23833
nrx = cx(1,lr);
// WRF:23834
z = zx(1,lr);
// WRF:23840
if((z>0.0f)) {
// WRF:23841
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:23842
for(i=1;i<=20;i+=1) {
// WRF:23843
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:23844
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:23845
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:23846
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:23847
}
// WRF:23850
if(((xv(1,il)>coeff_xvmx{}(il))||(((ioldlimiter==2)&&(xv(1,il)>lake_div(coeff_xvmx{}(il),8.f)))))) {
// WRF:23851
tmp = cx(1,il);
// WRF:23854
if((ioldlimiter==2)) {
// WRF:23855
x = lake_pow((lake_div(((6.f*rho0(1))*qx(1,il)),(((pi*xdn(1,il))*cx(1,il))))),(lake_div(1.f,3.f)));
// WRF:23856
x1 = lake_max(0.0e-3f,(x-3.0e-3f));
// WRF:23857
x2 = lake_max(0.5f,lake_div(x,6.0e-3f));
// WRF:23858
x3 = lake_pow(x2,3);
// WRF:23859
cx(1,il) = (cx(1,il)*lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:23860
xv(1,il) = lake_div(xv(1,il),lake_max(((1.f+(2.222e3f*lake_pow(x1,2)))),x3));
// WRF:23861
} else {
// WRF:23862
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:23863
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:23864
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:23865
}
// WRF:23870
if((tmp<cx(1,il))) {
// WRF:23872
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23873
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:23874
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23876
vr = xv(1,lr);
// WRF:23877
qr = qx(1,lr);
// WRF:23878
nrx = cx(1,lr);
// WRF:23879
z = zx(1,lr);
// WRF:23883
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:23884
for(i=1;i<=20;i+=1) {
// WRF:23885
if((lake_abs((alp-alpha(1,lr)))<0.01f)) {break;}
// WRF:23886
alpha(1,lr) = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:23887
alp = (lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((z*lake_pow(pi,2))))-1.f);
// WRF:23888
alp = lake_max(rnumin,lake_min(rnumax,alp));
// WRF:23889
}
// WRF:23892
}
// WRF:23893
}
// WRF:23899
g1 = lake_div((36.f*((alpha(1,lr)+2.0f))),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23900
if((true&&((((alpha(1,il)<=rnumin)||(alp==rnumin))||(alp==rnumax))))) {
// WRF:23902
if((rescale_high_alpha&&(alp>=(rnumax-0.01f)))) {
// WRF:23903
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(1.f,(xdn(1,il)))),2));
// WRF:23904
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:23906
} else if((rescale_low_alphar&&(alp<=rnumin))) {
// WRF:23907
z = lake_div((((36.f*((alpha(1,lr)+2.0f)))*nrx)*lake_pow(vr,2)),((((alpha(1,lr)+1.0f))*lake_pow(pi,2))));
// WRF:23908
zx(1,il) = z;
// WRF:23909
an(1,jy,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23910
}
// WRF:23911
}
// WRF:23915
}
// WRF:23916
}
// WRF:23918
}
// WRF:23920
}
// WRF:23924
} else if(((((il==lh)||(il==lhl))||(il==lf))||(((il==lr)&&(imurain==1))))) {
// WRF:23928
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:23930
if(((lf>1)&&(il==lf))) {
// WRF:23931
lfsave(1,5) = an(1,jy,1,coeff_ln{}(il));
// WRF:23932
lfsave(1,6) = cx(1,il);
// WRF:23933
}
// WRF:23935
if(((il==lhl)&&(lnhlf>1))) {
// WRF:23936
if((cx(1,lhl)>cxmin)) {
// WRF:23937
frac = lake_div(chxf(1,lhl),cx(1,lhl));
// WRF:23938
} else {
// WRF:23939
frac = 0.0f;
// WRF:23940
}
// WRF:23941
}
// WRF:23943
if(((il==lh)&&(lnhf>1))) {
// WRF:23944
if((cx(1,lh)>cxmin)) {
// WRF:23945
frach = lake_div(chxf(1,lh),cx(1,lh));
// WRF:23946
} else {
// WRF:23947
frach = 0.0f;
// WRF:23948
}
// WRF:23949
}
// WRF:23953
if((((iresetmoments==1)||(iresetmoments==il))||(iresetmoments==(-1)))) {
// WRF:23954
if((zx(1,il)<=zxmin)) {
// WRF:23956
qx(1,il) = 0.0f;
// WRF:23957
cx(1,il) = 0.0f;
// WRF:23958
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:23959
an(1,jgs,1,il) = qx(1,il);
// WRF:23960
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:23961
} else if(((iresetmoments==(-1))&&(qx(1,il)<coeff_qxmin{}(il)))) {
// WRF:23962
zx(1,il) = 0.0f;
// WRF:23963
cx(1,il) = 0.0f;
// WRF:23964
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:23966
qx(1,il) = 0.0f;
// WRF:23967
an(1,jgs,1,il) = qx(1,il);
// WRF:23968
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:23969
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23971
} else if(((cx(1,il)<=cxmin)&&(iresetmoments!=(-1)))) {
// WRF:23972
qx(1,lv) = (qx(1,lv)+qx(1,il));
// WRF:23973
zx(1,il) = 0.0f;
// WRF:23974
qx(1,il) = 0.0f;
// WRF:23975
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:23976
an(1,jgs,1,il) = qx(1,il);
// WRF:23977
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23978
}
// WRF:23979
} else {
// WRF:23980
if((zx(1,il)<0.0f)) {
// WRF:23981
zx(1,il) = 0.0f;
// WRF:23982
}
// WRF:23983
}
// WRF:23986
if(((zx(1,il)<=zxmin)&&(cx(1,il)<=cxmin))) {
// WRF:23987
zx(1,il) = 0.0f;
// WRF:23988
cx(1,il) = 0.0f;
// WRF:23989
an(1,jgs,1,lv) = (an(1,jgs,1,lv)+an(1,jgs,1,il));
// WRF:23990
qx(1,il) = 0.0f;
// WRF:23991
an(1,jgs,1,il) = qx(1,il);
// WRF:23992
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:23993
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:23994
}
// WRF:23996
if((qx(1,il)>coeff_qxmin{}(il))) {
// WRF:23998
xv(1,il) = lake_div((rho0(1)*qx(1,il)),((xdn(1,il)*lake_max(1.0e-9f,cx(1,il)))));
// WRF:23999
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:24001
if((xv(1,il)<coeff_xvmn{}(il))) {
// WRF:24002
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:24003
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:24004
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:24005
}
// WRF:24007
if(((zx(1,il)>0.0f)&&(cx(1,il)<=0.0f))) {
// WRF:24009
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:24011
z = zx(1,il);
// WRF:24012
qr = qx(1,il);
// WRF:24014
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6.f*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:24017
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)>0.0f))) {
// WRF:24021
chw = cx(1,il);
// WRF:24022
qr = qx(1,il);
// WRF:24025
g1 = lake_div(((((6.0f+alphamax))*((5.0f+alphamax)))*((4.0f+alphamax))),(((((3.0f+alphamax))*((2.0f+alphamax)))*((1.0f+alphamax)))));
// WRF:24027
zx(1,il) = lake_max((zxmin*1.1f),lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6*qr)),2)),((chw*lake_pow(((pi*xdn(1,il))),2)))));
// WRF:24028
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:24030
} else if(((zx(1,il)<=zxmin)&&(cx(1,il)<=0.0f))) {
// WRF:24037
zx(1,il) = (lake_div(1.e-19f,0.224f)*lake_pow((lake_div(coeff_xdn0{}(lr),coeff_xdn0{}(il))),2));
// WRF:24038
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:24040
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:24042
z = zx(1,il);
// WRF:24043
qr = qx(1,il);
// WRF:24045
cx(1,il) = lake_div(((g1*lake_pow(dn(1,jy,1),2))*lake_pow(((6.f*qr)),2)),((z*lake_pow(((pi*xdn(1,il))),2))));
// WRF:24046
an(1,jgs,1,coeff_ln{}(il)) = cx(1,il);
// WRF:24050
} else {
// WRF:24052
chw = cx(1,il);
// WRF:24053
qr = qx(1,il);
// WRF:24054
z = zx(1,il);
// WRF:24056
if((zx(1,il)>0.f)) {
// WRF:24059
rdi = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((rho0(1)*qr)),2)));
// WRF:24063
alp = (lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:24066
for(i=1;i<=10;i+=1) {
// WRF:24068
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:24069
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:24072
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:24075
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:24076
}
// WRF:24080
if((xv(1,il)>coeff_xvmx{}(il))) {
// WRF:24081
tmp = cx(1,il);
// WRF:24084
xv(1,il) = lake_min(coeff_xvmx{}(il),lake_max(coeff_xvmn{}(il),xv(1,il)));
// WRF:24085
xmas(1,il) = (xv(1,il)*xdn(1,il));
// WRF:24086
cx(1,il) = lake_div((rho0(1)*qx(1,il)),(xmas(1,il)));
// WRF:24087
if((tmp<cx(1,il))) {
// WRF:24088
g1 = lake_div((((36.f*((6.0f+alpha(1,il))))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),((((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))*lake_pow(pi,2))));
// WRF:24090
zx(1,il) = (zx(1,il)+((g1*lake_pow((lake_div(rho0(1),xdn(1,il))),2))*((lake_pow((lake_div(qx(1,il),tmp)),2)*((tmp-cx(1,il)))))));
// WRF:24091
an(1,jgs,1,coeff_lz{}(il)) = zx(1,il);
// WRF:24093
chw = cx(1,il);
// WRF:24094
qr = qx(1,il);
// WRF:24095
z = zx(1,il);
// WRF:24097
rdi = lake_div(((z*lake_pow(((lake_div(pi,6.f)*xdn(1,il))),2))*chw),(lake_pow(((rho0(1)*qr)),2)));
// WRF:24098
alp = (lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:24100
for(i=1;i<=10;i+=1) {
// WRF:24101
if((lake_abs((alp-alpha(1,il)))<0.01f)) {break;}
// WRF:24102
alpha(1,il) = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:24103
alp = (lake_div(((((6.f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*rdi)))-1.0f);
// WRF:24105
alp = lake_max(alphamin,lake_min(alphamax,alp));
// WRF:24106
}
// WRF:24109
}
// WRF:24110
}
// WRF:24116
g1 = lake_div(((((6.0f+alpha(1,il)))*((5.0f+alpha(1,il))))*((4.0f+alpha(1,il)))),(((((3.0f+alpha(1,il)))*((2.0f+alpha(1,il))))*((1.0f+alpha(1,il))))));
// WRF:24119
if((((lrescalelow(il)||rescale_high_alpha))&&((((alpha(1,il)<=alphamin)||(alp==alphamin))||(alp==alphamax))))) {
// WRF:24122
if((rescale_high_alpha&&(alp>=(alphamax-0.01f)))) {
// WRF:24123
cx(1,il) = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:24124
an(1,jy,1,coeff_ln{}(il)) = cx(1,il);
// WRF:24126
} else if((((lrescalelow(il)&&(alp<=alphamin))&&(!(((il==lh)&&(icvhl2h>0)))))&&(!(((il==lr)&&(!rescale_low_alphar)))))) {
// WRF:24129
wtest = false;
// WRF:24130
if((irescalerainopt==0)) {
// WRF:24131
wtest = false;
// WRF:24132
} else if((irescalerainopt==1)) {
// WRF:24133
wtest = (qx(1,lc)>coeff_qxmin{}(lc));
// WRF:24134
} else if((irescalerainopt==2)) {
// WRF:24135
wtest = ((qx(1,lc)>coeff_qxmin{}(lc))&&(wvel(1)<rescale_wthresh));
// WRF:24136
} else if((irescalerainopt==3)) {
// WRF:24137
wtest = (((temcg(1)>rescale_tempthresh)&&(qx(1,lc)>coeff_qxmin{}(lc)))&&(wvel(1)<rescale_wthresh));
// WRF:24138
}
// WRF:24140
if(((il==lr)&&((wtest||(!rescale_low_alphar))))) {
// WRF:24143
chw = (lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),z)*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:24144
cx(1,il) = chw;
// WRF:24145
an(1,jy,1,coeff_ln{}(il)) = chw;
// WRF:24146
} else {
// WRF:24148
z1 = lake_div((((g1*lake_pow(dn(1,jy,1),2))*(qr))*qr),chw);
// WRF:24149
z = (z1*lake_pow((lake_div(6.f,((pi*xdn(1,il))))),2));
// WRF:24150
zx(1,il) = z;
// WRF:24151
an(1,jy,1,coeff_lz{}(il)) = z;
// WRF:24152
}
// WRF:24158
}
// WRF:24160
}
// WRF:24163
}
// WRF:24166
}
// WRF:24170
}
// WRF:24172
if((lzr>1)) {
// WRF:24173
alpha2d(1,1,1) = lake_max(alphamin,lake_min(alphamax,alpha(1,lr)));
// WRF:24174
}
// WRF:24175
if((lzh>1)) {
// WRF:24176
alpha2d(1,1,2) = lake_max(alphamin,lake_min(alphamax,alpha(1,lh)));
// WRF:24177
}
// WRF:24178
if((lzhl>1)) {
// WRF:24179
alpha2d(1,1,3) = lake_max(alphamin,lake_min(alphamax,alpha(1,lhl)));
// WRF:24180
}
// WRF:24182
if(((il==lhl)&&(lnhlf>1))) {
// WRF:24184
chxf(1,lhl) = (frac*cx(1,lhl));
// WRF:24185
}
// WRF:24186
if(((il==lh)&&(lnhf>1))) {
// WRF:24188
chxf(1,lh) = (frach*cx(1,lh));
// WRF:24189
}
// WRF:24199
}
// WRF:24209
}
// WRF:24211
}
// WRF:24212
}
// WRF:24214
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:24216
if((il==lh)) {
// WRF:24217
if((lnhf>1)) {
// WRF:24218
an(1,jy,1,lnhf) = lake_max(chxf(1,lh),0.0f);
// WRF:24219
}
// WRF:24220
}
// WRF:24222
if((il==lhl)) {
// WRF:24224
if((lnhlf>1)) {
// WRF:24226
an(1,jy,1,lnhlf) = lake_max(chxf(1,lhl),0.0f);
// WRF:24227
}
// WRF:24228
}
// WRF:24229
an(1,jy,1,coeff_ln{}(il)) = lake_max(cx(1,il),0.0f);
// WRF:24230
}
// WRF:24231
}
// WRF:24232
}
// WRF:24234
if((lcin>1)) {
// WRF:24235
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:24236
an(1,jy,1,lcin) = lake_max(0.0f,ccin(1));
// WRF:24237
}
// WRF:24238
}
// WRF:24240
if((ipconc>=2)) {
// WRF:24241
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:24242
if((lss>1)) {
// WRF:24243
an(1,jy,1,lss) = lake_max(0.0f,ssmax(1));
// WRF:24244
}
// WRF:24246
if((lccn>1)) {
// WRF:24247
an(1,jy,1,lccn) = lake_max(0.0f,ccnc(1));
// WRF:24248
}
// WRF:24249
}
// WRF:24250
}
// WRF:24252
} else if(((ipconc==0)&&(lni>1))) {
// WRF:24254
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:24255
an(1,jy,1,lni) = lake_max(cx(1,li),0.0f);
// WRF:24256
}
// WRF:24259
}
// WRF:24261
if(ldovol) {
// WRF:24263
#pragma unroll
for(il=li;il<=lhab;il+=1) {
// WRF:24265
if((coeff_lvol{}(il)>=1)) {
// WRF:24267
for(int _point_once=0;_point_once<(1);++_point_once) {
// WRF:24269
an(1,jy,1,coeff_lvol{}(il)) = lake_max(0.0f,vx(1,il));
// WRF:24270
}
// WRF:24272
}
// WRF:24274
}
// WRF:24276
}
// WRF:24282
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:24286
if((ndebug>0)) {; /* native diagnostic only */}
// WRF:24288
L9998: ;
;
return;
}
L1000: ;
;
// WRF:24308
L1200: ;
;
// WRF:24314
return;
}
__device__ __forceinline__ void nucond(int nx,int ny,int nz,int na,int jyslab,int nor,int norz,float dtp,int nxi,Array<const float> dz3d,Array<float> t0,Array<const float> t9,Array<float> an,Array<const float> dn,Array<const float> p2,Array<const float> pn,Array<const float> w,int ngs,Array<const float> axtra,bool io_flag,Array<const float> ssfilt,Array<const float> t00,Array<const float> t77,bool flag_qndrop) {
constexpr int ng1=1;
FixedStorage<float,((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(NZ+0)> pb{};
FixedStorage<float,((NZ+0)-((-0)+ng1)+1),((-0)+ng1),(NZ+0)> pinit{};
float dv;
float ccnefactwo;
float sstmp;
float cn1;
float cnuctmp;
const float cwmas30=((1000.f*0.523599f)*lake_pow(((2.f*30.e-6f)),3));
const float cwmas20=((1000.f*0.523599f)*lake_pow(((2.f*20.e-6f)),3));
int nxmpb;
int nzmpb;
int nxz;
int mgs;
int numgs;
int inumgs;
int ngscnt;
FixedStorage<int,(1-1+1),1,1> igs{};
FixedStorage<int,(1-1+1),1,1> kgs{};
FixedStorage<int,(1-1+1),1,1> kgsp{};
FixedStorage<int,(1-1+1),1,1> kgsm{};
int nsvcnt;
int ix;
int kz;
int i;
int n;
int kp1;
int km1;
int jy;
int jgs;
int ixb;
int ixe;
int jyb;
int jye;
int kzb;
int kze;
int itile;
int jtile;
int ktile;
int ixend;
int jyend;
int kzend;
int kzbeg;
int nxend;
int nyend;
int nzend;
int nzbeg;
FixedStorage<float,(1-1+1),1,1> ccnc{};
FixedStorage<float,(1-1+1),1,1> ccna{};
FixedStorage<float,(1-1+1),1,1> cnuc{};
FixedStorage<float,(1-1+1),1,1> cwnccn{};
FixedStorage<float,(1-1+1),1,1> ccnc_nu{};
FixedStorage<float,(1-1+1),1,1> ccnc_ac{};
FixedStorage<float,(1-1+1),1,1> ccnc_co{};
FixedStorage<float,(1-1+1),1,1> ccncuf{};
const float sscb=2.0f;
constexpr int idecss=1;
constexpr int iba=1;
constexpr int ifilt=0;
float temp1;
float temp2;
FixedStorage<float,(1-1+1),1,1> ssmax{};
float ssmx;
float dnnet;
float dqnet;
FixedStorage<float,(1-1+1),1,1> ventrx{};
FixedStorage<float,(1-1+1),1,1> ventrxn{};
float volb;
float t2s;
const float aa1=9.44e15f;
const float aa2=5.78e3f;
float ec0;
float ex1;
float ft;
FixedStorage<float,(1-1+1),1,1> rhoinv{};
float chw;
float g1;
float rd1;
float ac1;
float bc;
float taus;
float c1;
float d1;
float e1;
float f1;
float p380;
float tmp;
float tmp2;
float tmpmx;
float fw;
float qctmp;
float x;
float y;
float del;
float r;
float alpr;
double vent1;
double vent2;
float g1palp;
float bs;
float v1;
float v2;
float d1r;
float d1i;
float d1s;
float e1i;
int nc;
float dtcon;
float dtcon1;
float dtcon2;
float delta;
int ltemq1;
int ltemq1m;
float dqv;
float qv1;
float ss1;
float ss2;
float qvs1;
float dqvs;
float dtemp;
float dt1;
float ssi1;
float ssi2;
float dqvi;
float dqvis;
float dqvii;
float qis1;
float dqvr;
float dqc;
float dqr;
float dqi;
float dqs;
float qv1m;
float qvs1m;
float ss1m;
float ssi1m;
float qis1m;
float cwmastmp;
float dcloud;
float dcloud2;
float dcrit;
FixedStorage<float,(1-1+1),1,1> cn{};
FixedStorage<float,(1-1+1),1,1> cnuf{};
float ccwmax;
int ltemq;
int il;
FixedStorage<float,(1-1+1),1,1> es{};
FixedStorage<float,(1-1+1),1,1> ssf{};
FixedStorage<float,(1-1+1),1,1> ssfkp1{};
FixedStorage<float,(1-1+1),1,1> ssfkm1{};
FixedStorage<float,(1-1+1),1,1> ssat0{};
const float ssfcut=4.0f;
FixedStorage<float,(1-1+1),1,1> ssfjp1{};
FixedStorage<float,(1-1+1),1,1> ssfjm1{};
FixedStorage<float,(1-1+1),1,1> ssfip1{};
FixedStorage<float,(1-1+1),1,1> ssfim1{};
const float supcb=0.5f;
const float supmx=238.0f;
float r2dxm;
float r2dym;
float r2dzm;
float dssdz;
float dssdy;
float dssdx;
const float epsi=0.622f;
const float d=0.266f;
float r1;
float qevap;
float vr;
float nrx;
float qr;
float z1;
float z2;
float rdi;
float alp;
float xnutmp;
float xnuc;
float ctmp;
float ccwtmp;
float f5;
float qvs0;
float t0p1;
float t0p3;
float qvex;
FixedStorage<float,(1-1+1),1,1> dqvcnd{};
FixedStorage<float,(1-1+1),1,1> dqwv{};
FixedStorage<float,(1-1+1),1,1> dqcw{};
FixedStorage<float,(1-1+1),1,1> dqci{};
FixedStorage<float,(1-1+1),1,1> temp{};
FixedStorage<float,(1-1+1),1,1> tempc{};
FixedStorage<float,(1-1+1),1,1> temg{};
FixedStorage<float,(1-1+1),1,1> temcg{};
FixedStorage<float,(1-1+1),1,1> theta{};
FixedStorage<float,(1-1+1),1,1> qvap{};
FixedStorage<float,(1-1+1),1,1> temgx{};
FixedStorage<float,(1-1+1),1,1> temcgx{};
FixedStorage<float,(1-1+1),1,1> qvs{};
FixedStorage<float,(1-1+1),1,1> qis{};
FixedStorage<float,(1-1+1),1,1> qss{};
FixedStorage<float,(1-1+1),1,1> pqs{};
FixedStorage<float,(1-1+1),1,1> felv{};
FixedStorage<float,(1-1+1),1,1> felf{};
FixedStorage<float,(1-1+1),1,1> fels{};
FixedStorage<float,(1-1+1),1,1> felvcp{};
FixedStorage<float,(1-1+1),1,1> felvpi{};
FixedStorage<float,(1-1+1),1,1> gamw{};
FixedStorage<float,(1-1+1),1,1> gams{};
FixedStorage<float,(1-1+1),1,1> tsqr{};
FixedStorage<float,(1-1+1),1,1> ssi{};
FixedStorage<float,(1-1+1),1,1> ssw{};
FixedStorage<float,(1-1+1),1,1> cc3{};
FixedStorage<float,(1-1+1),1,1> cqv1{};
FixedStorage<float,(1-1+1),1,1> cqv2{};
FixedStorage<float,(1-1+1),1,1> qcwtmp{};
float qtmp;
FixedStorage<float,(1-1+1),1,1> fvent{};
FixedStorage<float,(1-1+1),1,1> fwvdf{};
FixedStorage<float,(1-1+1),1,1> ftka{};
FixedStorage<float,(1-1+1),1,1> fthdf{};
FixedStorage<float,(1-1+1),1,1> fadvisc{};
FixedStorage<float,(1-1+1),1,1> fakvisc{};
FixedStorage<float,(1-1+1),1,1> fci{};
FixedStorage<float,(1-1+1),1,1> fcw{};
FixedStorage<float,(1-1+1),1,1> fschm{};
FixedStorage<float,(1-1+1),1,1> fpndl{};
FixedStorage<float,(1-1+1),1,1> pres{};
FixedStorage<float,(1-1+1),1,1> pipert{};
FixedStorage<float,(1-1+1),1,1> pk{};
FixedStorage<float,(1-1+1),1,1> rho0{};
FixedStorage<float,(1-1+1),1,1> pi0{};
FixedStorage<float,(1-1+1),1,1> rhovt{};
FixedStorage<float,(1-1+1),1,1> thetap{};
FixedStorage<float,(1-1+1),1,1> theta0{};
FixedStorage<float,(1-1+1),1,1> qwvp{};
FixedStorage<float,(1-1+1),1,1> qv0{};
FixedStorage<float,(1-1+1),1,1> thsave{};
FixedStorage<float,(1-1+1),1,1> qss0{};
FixedStorage<float,(1-1+1),1,1> fcqv1{};
FixedStorage<float,(1-1+1),1,1> wvel{};
FixedStorage<float,(1-1+1),1,1> wvelkm1{};
FixedStorage<float,(1-1+1),1,1> wvdf{};
FixedStorage<float,(1-1+1),1,1> tka{};
FixedStorage<float,(1-1+1),1,1> advisc{};
FixedStorage<float,(1-1+1),1,1> rwvent{};
FixedStorage<float,(1-1+1)*(lhab-lv+1),1,1,lv,lhab> qx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> cx{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xv{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xmas{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> xdn{};
FixedStorage<float,(1-1+1)*(lhab-lc+1)*(3-1+1),1,1,lc,lhab,1,3> xdia{};
FixedStorage<float,(1-1+1)*(lhab-lc+1),1,1,lc,lhab> alpha{};
FixedStorage<float,(1-1+1)*(lhab-lr+1),1,1,lr,lhab> zx{};
FixedStorage<bool,(lqmx-lc+1),lc,lqmx> zerocx{};
bool lprint;
constexpr int iunit=0;
float frac;
float hwdn;
float tmpg;
float cvm;
float cpm;
float rmm;
const float cpv=1885.0f;
int kstag;
int count;
// WRF:11719
jy = 1;
// WRF:11719
L2200: ;
;
// WRF:11732
if(true) {
// WRF:11734
frac = 1.0f;
// WRF:11740
for(kz=1;kz<=nz;kz+=1) {
// WRF:11742
for(ix=1;ix<=nxi;ix+=1) {
// WRF:11744
t0(ix,jy,kz) = (an(ix,jy,kz,lt)*t77(ix,jy,kz));
// WRF:11746
for(int _s0=lc;_s0<=lqmx;++_s0) {zerocx(_s0) = false;}
// WRF:11747
#pragma unroll
for(il=lc;il<=lhab;il+=1) {
// WRF:11748
if(((iresetmoments==1)||(iresetmoments==il))) {
// WRF:11749
if((coeff_ln{}(il)>1)) {zerocx(il) = ((an(ix,jy,kz,coeff_ln{}(il))<cxmin));}
// WRF:11750
if((coeff_lz{}(il)>1)) {zerocx(il) = ((zerocx(il)||(an(ix,jy,kz,coeff_lz{}(il))<zxmin)));}
// WRF:11751
} else {
// WRF:11752
if((il==lc)) {
// WRF:11753
if((coeff_ln{}(il)>1)) {zerocx(il) = (((an(ix,jy,kz,coeff_ln{}(il))<=0))&&(!flag_qndrop));}
// WRF:11754
} else {
// WRF:11755
if((coeff_ln{}(il)>1)) {zerocx(il) = ((an(ix,jy,kz,coeff_ln{}(il))<=0));}
// WRF:11756
}
// WRF:11757
}
// WRF:11758
}
// WRF:11760
if((lhl>1)) {
// WRF:11762
if((lzhl>1)) {
// WRF:11764
an(ix,jy,kz,lzhl) = lake_max(0.0f,an(ix,jy,kz,lzhl));
// WRF:11766
if(((an(ix,jy,kz,lhl)>=(frac*coeff_qxmin{}(lhl)))&&rescale_low_alpha)) {
// WRF:11768
if((an(ix,jy,kz,lnhl)>0.0f)) {
// WRF:11770
if((lvhl>1)) {
// WRF:11771
if((an(ix,jy,kz,lvhl)>0.0f)) {
// WRF:11772
hwdn = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),an(ix,jy,kz,lvhl));
// WRF:11773
} else {
// WRF:11774
hwdn = coeff_xdn0{}(lhl);
// WRF:11775
}
// WRF:11776
hwdn = lake_max(coeff_xdnmn{}(lhl),hwdn);
// WRF:11777
} else {
// WRF:11778
hwdn = coeff_xdn0{}(lhl);
// WRF:11779
}
// WRF:11781
chw = an(ix,jy,kz,lnhl);
// WRF:11782
g1 = lake_div(((((6.0f+alphamin))*((5.0f+alphamin)))*((4.0f+alphamin))),(((((3.0f+alphamin))*((2.0f+alphamin)))*((1.0f+alphamin)))));
// WRF:11784
z1 = lake_div((((g1*lake_pow(dn(ix,jy,kz),2))*(an(ix,jy,kz,lhl)))*an(ix,jy,kz,lhl)),chw);
// WRF:11785
z1 = (z1*lake_pow((lake_div(6.f,((pi*hwdn)))),2));
// WRF:11786
} else {
// WRF:11787
z1 = 0.0f;
// WRF:11788
}
// WRF:11790
an(ix,jy,kz,lzhl) = lake_min(z1,an(ix,jy,kz,lzhl));
// WRF:11792
if((an(ix,jy,kz,lnhl)<1.e-5f)) {
// WRF:11794
}
// WRF:11795
}
// WRF:11797
}
// WRF:11799
if(((an(ix,jy,kz,lhl)<(frac*coeff_qxmin{}(lhl)))||zerocx(lhl))) {
// WRF:11802
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,lhl));
// WRF:11803
an(ix,jy,kz,lhl) = 0.0f;
// WRF:11806
if((ipconc>=5)) {
// WRF:11807
an(ix,jy,kz,lnhl) = 0.0f;
// WRF:11808
}
// WRF:11810
if((lvhl>1)) {
// WRF:11811
an(ix,jy,kz,lvhl) = 0.0f;
// WRF:11812
}
// WRF:11814
if((lhlw>1)) {
// WRF:11815
an(ix,jy,kz,lhlw) = 0.0f;
// WRF:11816
}
// WRF:11818
if((lnhlf>1)) {
// WRF:11819
an(ix,jy,kz,lnhlf) = 0.0f;
// WRF:11820
}
// WRF:11822
if((lzhl>1)) {
// WRF:11823
an(ix,jy,kz,lzhl) = 0.0f;
// WRF:11824
}
// WRF:11826
} else {
// WRF:11827
if((coeff_lvol{}(lhl)>1)) {
// WRF:11828
if((an(ix,jy,kz,lvhl)>0.0f)) {
// WRF:11829
tmp = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),an(ix,jy,kz,lvhl));
// WRF:11830
} else {
// WRF:11831
tmp = rho_qhl;
// WRF:11832
an(ix,jy,kz,lvhl) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),tmp);
// WRF:11833
}
// WRF:11835
if((tmp<coeff_xdnmn{}(lhl))) {
// WRF:11836
tmp = lake_max(coeff_xdnmn{}(lhl),tmp);
// WRF:11837
an(ix,jy,kz,lvhl) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),tmp);
// WRF:11838
}
// WRF:11840
if(((tmp>coeff_xdnmx{}(lhl))&&(lhlw<=0))) {
// WRF:11841
tmp = lake_min(coeff_xdnmx{}(lhl),tmp);
// WRF:11842
an(ix,jy,kz,lvhl) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),tmp);
// WRF:11843
} else if(((tmp>coeff_xdnmx{}(lhl))&&(lhlw>1))) {
// WRF:11844
fw = lake_div(an(ix,jy,kz,lhlw),an(ix,jy,kz,lhl));
// WRF:11849
tmpmx = lake_div(coeff_xdnmx{}(lhl),((1.f-(fw*((1.f-lake_div(coeff_xdnmx{}(lhl),coeff_xdnmx{}(lr))))))));
// WRF:11851
if((tmp>tmpmx)) {
// WRF:11852
an(ix,jy,kz,lvhl) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),tmpmx);
// WRF:11853
}
// WRF:11862
}
// WRF:11864
if((lhlw>1)) {
// WRF:11865
if((an(ix,jy,kz,lhlw)>(0.98f*an(ix,jy,kz,lhl)))) {
// WRF:11866
tmp = coeff_xdnmx{}(lr);
// WRF:11867
an(ix,jy,kz,lvhl) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),tmp);
// WRF:11868
}
// WRF:11869
}
// WRF:11871
}
// WRF:11875
if((((((ipconc==5)&&(an(ix,jy,kz,lhl)>coeff_qxmin{}(lhl)))&&(alphahl<=0.1f))&&(lnhl>1))&&(lzhl==0))) {
// WRF:11877
if((lvhl>1)) {
// WRF:11878
hwdn = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lhl)),an(ix,jy,kz,lvhl));
// WRF:11879
} else {
// WRF:11880
hwdn = coeff_xdn0{}(lhl);
// WRF:11881
}
// WRF:11882
tmp = lake_div(((hwdn*an(ix,jy,kz,lnhl))),((dn(ix,jy,kz)*an(ix,jy,kz,lhl))));
// WRF:11883
tmpg = (an(ix,jy,kz,lnhl)*lake_pow(((tmp*(3.14159f))),(lake_div(1.f,3.f))));
// WRF:11884
if((tmpg<cnohlmn)) {
// WRF:11885
tmp = lake_pow(((lake_div((hwdn),((dn(ix,jy,kz)*an(ix,jy,kz,lhl))))*(3.14159f))),(lake_div(1.f,3.f)));
// WRF:11886
an(ix,jy,kz,lnhl) = lake_pow((lake_div(cnohlmn,tmp)),(lake_div(3.f,4.f)));
// WRF:11887
}
// WRF:11889
}
// WRF:11892
}
// WRF:11894
}
// WRF:11898
if((lzh>1)) {
// WRF:11900
an(ix,jy,kz,lzh) = lake_max(0.0f,an(ix,jy,kz,lzh));
// WRF:11902
if(((false&&(an(ix,jy,kz,lh)>=(frac*coeff_qxmin{}(lh))))&&rescale_low_alpha)) {
// WRF:11904
if((an(ix,jy,kz,lnh)>0.0f)) {
// WRF:11906
if((lvh>1)) {
// WRF:11907
if((an(ix,jy,kz,lvh)>0.0f)) {
// WRF:11908
hwdn = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),an(ix,jy,kz,lvh));
// WRF:11909
} else {
// WRF:11910
hwdn = coeff_xdn0{}(lh);
// WRF:11911
}
// WRF:11912
hwdn = lake_max(coeff_xdnmn{}(lh),hwdn);
// WRF:11913
} else {
// WRF:11914
hwdn = coeff_xdn0{}(lh);
// WRF:11915
}
// WRF:11917
chw = an(ix,jy,kz,lnh);
// WRF:11918
g1 = lake_div(((((6.0f+alphamin))*((5.0f+alphamin)))*((4.0f+alphamin))),(((((3.0f+alphamin))*((2.0f+alphamin)))*((1.0f+alphamin)))));
// WRF:11920
z1 = lake_div((((g1*lake_pow(dn(ix,jy,kz),2))*(an(ix,jy,kz,lh)))*an(ix,jy,kz,lh)),chw);
// WRF:11921
z1 = (z1*lake_pow((lake_div(6.f,((pi*hwdn)))),2));
// WRF:11922
} else {
// WRF:11923
z1 = 0.0f;
// WRF:11924
}
// WRF:11926
an(ix,jy,kz,lzh) = lake_min(z1,an(ix,jy,kz,lzh));
// WRF:11928
if((an(ix,jy,kz,lnh)<1.e-5f)) {
// WRF:11930
}
// WRF:11931
}
// WRF:11933
}
// WRF:11935
if(((an(ix,jy,kz,lh)<(frac*coeff_qxmin{}(lh)))||zerocx(lh))) {
// WRF:11938
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,lh));
// WRF:11939
an(ix,jy,kz,lh) = 0.0f;
// WRF:11942
if((ipconc>=5)) {
// WRF:11943
an(ix,jy,kz,lnh) = 0.0f;
// WRF:11944
}
// WRF:11946
if((lvh>1)) {
// WRF:11947
an(ix,jy,kz,lvh) = 0.0f;
// WRF:11948
}
// WRF:11950
if((lhw>1)) {
// WRF:11951
an(ix,jy,kz,lhw) = 0.0f;
// WRF:11952
}
// WRF:11954
if((lnhf>1)) {
// WRF:11955
an(ix,jy,kz,lnhf) = 0.0f;
// WRF:11956
}
// WRF:11958
if((lzh>1)) {
// WRF:11959
an(ix,jy,kz,lzh) = 0.0f;
// WRF:11960
}
// WRF:11962
} else {
// WRF:11963
if((coeff_lvol{}(lh)>1)) {
// WRF:11964
if((an(ix,jy,kz,lvh)>0.0f)) {
// WRF:11965
tmp = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),an(ix,jy,kz,lvh));
// WRF:11966
} else {
// WRF:11967
tmp = rho_qh;
// WRF:11968
an(ix,jy,kz,lvh) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),tmp);
// WRF:11969
}
// WRF:11971
if((tmp<coeff_xdnmn{}(lh))) {
// WRF:11972
tmp = lake_max(coeff_xdnmn{}(lh),tmp);
// WRF:11973
an(ix,jy,kz,lvh) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),tmp);
// WRF:11974
}
// WRF:11976
if(((tmp>coeff_xdnmx{}(lh))&&(lhw<=0))) {
// WRF:11977
tmp = lake_min(coeff_xdnmx{}(lh),tmp);
// WRF:11978
an(ix,jy,kz,lvh) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),tmp);
// WRF:11979
} else if(((tmp>coeff_xdnmx{}(lh))&&(lhw>1))) {
// WRF:11980
fw = lake_div(an(ix,jy,kz,lhw),an(ix,jy,kz,lh));
// WRF:11984
tmpmx = lake_div(coeff_xdnmx{}(lh),((1.f-(fw*((1.f-lake_div(coeff_xdnmx{}(lh),coeff_xdnmx{}(lr))))))));
// WRF:11986
if((tmp>tmpmx)) {
// WRF:11987
an(ix,jy,kz,lvh) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),tmpmx);
// WRF:11988
}
// WRF:11998
}
// WRF:12000
if((lhw>1)) {
// WRF:12001
if((an(ix,jy,kz,lhw)>(0.98f*an(ix,jy,kz,lh)))) {
// WRF:12002
tmp = coeff_xdnmx{}(lr);
// WRF:12003
an(ix,jy,kz,lvh) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),tmp);
// WRF:12004
}
// WRF:12005
}
// WRF:12007
}
// WRF:12010
if((((((ipconc==5)&&(an(ix,jy,kz,lh)>coeff_qxmin{}(lh)))&&(alphah<=0.1f))&&(lnh>1))&&(lzh==0))) {
// WRF:12012
if((lvh>1)) {
// WRF:12013
if((an(ix,jy,kz,lvh)>0.0f)) {
// WRF:12014
hwdn = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,lh)),an(ix,jy,kz,lvh));
// WRF:12015
} else {
// WRF:12016
hwdn = coeff_xdn0{}(lh);
// WRF:12017
}
// WRF:12018
hwdn = lake_max(coeff_xdnmn{}(lh),hwdn);
// WRF:12019
} else {
// WRF:12020
hwdn = coeff_xdn0{}(lh);
// WRF:12021
}
// WRF:12022
tmp = lake_div(((hwdn*an(ix,jy,kz,lnh))),((dn(ix,jy,kz)*an(ix,jy,kz,lh))));
// WRF:12023
tmpg = (an(ix,jy,kz,lnh)*lake_pow(((tmp*(3.14159f))),(lake_div(1.f,3.f))));
// WRF:12024
if((tmpg<cnohmn)) {
// WRF:12027
tmp = lake_pow(((lake_div((hwdn),((dn(ix,jy,kz)*an(ix,jy,kz,lh))))*(3.14159f))),(lake_div(1.f,3.f)));
// WRF:12028
an(ix,jy,kz,lnh) = lake_pow((lake_div(cnohmn,tmp)),(lake_div(3.f,4.f)));
// WRF:12029
}
// WRF:12031
}
// WRF:12033
}
// WRF:12036
if(((an(ix,jy,kz,ls)<(frac*coeff_qxmin{}(ls)))||zerocx(ls))) {
// WRF:12038
if((t0(ix,jy,kz)<273.15f)) {
// WRF:12040
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,ls));
// WRF:12041
an(ix,jy,kz,ls) = 0.0f;
// WRF:12044
if((ipconc>=4)) {
// WRF:12046
an(ix,jy,kz,lns) = 0.0f;
// WRF:12047
}
// WRF:12049
if((lvs>1)) {
// WRF:12050
an(ix,jy,kz,lvs) = 0.0f;
// WRF:12051
}
// WRF:12053
if((lsw>1)) {
// WRF:12054
an(ix,jy,kz,lsw) = 0.0f;
// WRF:12055
}
// WRF:12057
} else {
// WRF:12059
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,ls));
// WRF:12060
an(ix,jy,kz,ls) = 0.0f;
// WRF:12063
if((lvs>1)) {
// WRF:12064
an(ix,jy,kz,lvs) = 0.0f;
// WRF:12065
}
// WRF:12067
if((lsw>1)) {
// WRF:12068
an(ix,jy,kz,lsw) = 0.0f;
// WRF:12069
}
// WRF:12071
if((ipconc>=4)) {
// WRF:12073
an(ix,jy,kz,lns) = 0.0f;
// WRF:12074
}
// WRF:12076
}
// WRF:12079
} else if((coeff_lvol{}(ls)>1)) {
// WRF:12080
if((an(ix,jy,kz,lvs)>0.0f)) {
// WRF:12081
tmp = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,ls)),an(ix,jy,kz,lvs));
// WRF:12082
if(((tmp>coeff_xdnmx{}(ls))||(tmp<coeff_xdnmn{}(ls)))) {
// WRF:12083
tmp = lake_min(coeff_xdnmx{}(ls),lake_max(coeff_xdnmn{}(ls),tmp));
// WRF:12084
an(ix,jy,kz,lvs) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,ls)),tmp);
// WRF:12085
}
// WRF:12086
} else {
// WRF:12087
tmp = rho_qs;
// WRF:12088
an(ix,jy,kz,lvs) = lake_div((dn(ix,jy,kz)*an(ix,jy,kz,ls)),tmp);
// WRF:12089
}
// WRF:12092
}
// WRF:12094
if((lzr>1)) {
// WRF:12095
an(ix,jy,kz,lzr) = lake_max(0.0f,an(ix,jy,kz,lzr));
// WRF:12096
}
// WRF:12098
if(((an(ix,jy,kz,lr)<(frac*coeff_qxmin{}(lr)))||zerocx(lr))) {
// WRF:12100
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,lr));
// WRF:12101
an(ix,jy,kz,lr) = 0.0f;
// WRF:12102
if((ipconc>=3)) {
// WRF:12104
an(ix,jy,kz,lnr) = 0.0f;
// WRF:12105
}
// WRF:12107
if((lzr>1)) {
// WRF:12108
an(ix,jy,kz,lzr) = 0.0f;
// WRF:12109
}
// WRF:12111
}
// WRF:12116
if(((an(ix,jy,kz,li)<=(frac*coeff_qxmin{}(li)))||zerocx(li))) {
// WRF:12118
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,li));
// WRF:12119
an(ix,jy,kz,li) = 0.0f;
// WRF:12120
if((ipconc>=1)) {
// WRF:12121
an(ix,jy,kz,lni) = 0.0f;
// WRF:12122
}
// WRF:12123
}
// WRF:12128
if((lis>1)) {
// WRF:12129
if(((an(ix,jy,kz,lis)<=(frac*coeff_qxmin{}(lis)))||zerocx(lis))) {
// WRF:12131
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,lis));
// WRF:12132
an(ix,jy,kz,lis) = 0.0f;
// WRF:12133
if((ipconc>=1)) {
// WRF:12134
an(ix,jy,kz,lnis) = 0.0f;
// WRF:12135
}
// WRF:12137
} else if((icespheres>=2)) {
// WRF:12138
km1 = lake_max(1,(kz-1));
// WRF:12139
if((((((0.5f*((w(ix,jy,kz)+w(ix,jy,(kz+1)))))<(-1.0f))||(((icespheres==3)&&(((t0(ix,jy,kz)<232.15f)||(an(ix,jy,kz,lc)<coeff_qxmin{}(lc)))))))||(((icespheres==5)&&(((t0(ix,jy,kz)<232.15f)||(((an(ix,jy,kz,lc)<coeff_qxmin{}(lc))&&(an(ix,jy,km1,lc)<coeff_qxmin{}(lc)))))))))||(((icespheres==4)&&((t0(ix,jy,kz)<235.15f)))))) {
// WRF:12144
an(ix,jy,kz,li) = (an(ix,jy,kz,li)+an(ix,jy,kz,lis));
// WRF:12145
an(ix,jy,kz,lni) = (an(ix,jy,kz,lni)+an(ix,jy,kz,lnis));
// WRF:12146
an(ix,jy,kz,lis) = 0.0f;
// WRF:12147
an(ix,jy,kz,lnis) = 0.0f;
// WRF:12149
}
// WRF:12151
}
// WRF:12152
}
// WRF:12158
if(((an(ix,jy,kz,lc)<=(frac*coeff_qxmin{}(lc)))||zerocx(lc))) {
// WRF:12160
an(ix,jy,kz,lv) = (an(ix,jy,kz,lv)+an(ix,jy,kz,lc));
// WRF:12161
an(ix,jy,kz,lc) = 0.0f;
// WRF:12162
if((ipconc>=2)) {
// WRF:12163
if(((lccn>1)||(ac_opt==1))) {
// WRF:12164
if(((irenuc<5)&&(lccna<=1))) {
// WRF:12165
if((ac_opt==0)) {
// WRF:12166
an(ix,jy,kz,lccn) = (an(ix,jy,kz,lccn)+lake_max(0.0f,an(ix,jy,kz,lnc)));
// WRF:12167
}
// WRF:12168
} else if((lccna>1)) {
// WRF:12169
an(ix,jy,kz,lccna) = lake_max(0.0f,(an(ix,jy,kz,lccna)-lake_max(0.0f,an(ix,jy,kz,lnc))));
// WRF:12170
}
// WRF:12171
}
// WRF:12172
an(ix,jy,kz,lnc) = 0.0f;
// WRF:12173
if((lccn>1)) {an(ix,jy,kz,lccn) = lake_max(0.0f,an(ix,jy,kz,lccn));}
// WRF:12175
if(((lccna>0)&&(ac_opt==0))) {
// WRF:12176
if(restoreccn) {
// WRF:12177
tmp = (an(ix,jy,kz,li)+an(ix,jy,kz,ls));
// WRF:12179
if(((an(ix,jy,kz,lccna)>1.f)&&(tmp<coeff_qxmin{}(li)))) {an(ix,jy,kz,lccna) = (an(ix,jy,kz,lccna)*gfk_exp(lake_div((-dtp),ccntimeconst)));}
// WRF:12180
}
// WRF:12181
} else if((((lccn>1)&&restoreccn)&&(ac_opt==0))) {
// WRF:12183
tmp = (an(ix,jy,kz,li)+an(ix,jy,kz,ls));
// WRF:12188
if((((an(ix,jy,kz,lccn)>1.f)&&(tmp<coeff_qxmin{}(li)))&&(((an(ix,jy,kz,lccn)<(dn(ix,jy,kz)*qccn))||(!invertccn))))) {
// WRF:12192
an(ix,jy,kz,lccn) = ((dn(ix,jy,kz)*qccn)-(lake_max(0.0f,((dn(ix,jy,kz)*qccn)-an(ix,jy,kz,lccn)))*gfk_exp(lake_div((-dtp),ccntimeconst))));
// WRF:12194
}
// WRF:12196
}
// WRF:12198
}
// WRF:12200
}
// WRF:12202
}
// WRF:12204
}
// WRF:12206
}
// WRF:12208
if((ndebug>=1)) {; /* native diagnostic only */}
// WRF:12213
L9999: ;
return;
}
};
}
