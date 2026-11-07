"""MPAS-A v8.4.2 longwave driver and audited RRTMG CUDA solver.

MPAS: physics_wrf/module_ra_rrtmg_lw.F:1-12538; driver slice:
mpas_atmphys_driver_radiation_lw.F:297-719,865-885.
The active pinned source compiles RKIND and kind_rb as IEEE binary32.
Gas optics/solver reuse is justified by receipts/source-audit.json and
mpas-vs-local-wrf461.diff in the p22 delivery (WRF CRLF SHA cefe75fa...).
Legacy engine code and defaults are never changed. Inputs and coefficient
arrays are device resident; Python only allocates and launches kernels.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
import numpy as np
from woof.core import rrtmg_lw as _legacy

GPU_FSLOTS = _legacy.GPU_FSLOTS
GPU_ISLOTS = _legacy.GPU_ISLOTS
NBNDLW, NGPTLW, MXMOL, MAXXSEC = 16, 140, 38, 4
MAX_RADIATION_LAYERS = _legacy.MAX_RADIATION_LAYERS
WTDIFF = _legacy.WTDIFF
LWBatchScratch = _legacy.LWBatchScratch
_MODULES = {}

# MPAS: module_ra_rrtmg_lw.F:1712-2030. Unsigned arithmetic preserves the
# Fortran int32 bit operations and multiplication wrap without C++ signed UB.
_RANDOM_SOURCE = r"""
struct mpas_mt_sequence { int current; unsigned int state[624]; };
// MPAS: module_ra_rrtmg_lw.F:1755-1785 (mixbits, twist, nextState).
__device__ unsigned int mpas_mt_twist(unsigned int u,unsigned int v) {
    unsigned int mix=(u&0x80000000u)|(v&0x7fffffffu);
    return (mix>>1)^((v&1u)?0x9908b0dfu:0u);
}
// MPAS: module_ra_rrtmg_lw.F:1769-1787.
__device__ void mpas_mt_next(mpas_mt_sequence &s) {
    for(int k=0;k<227;++k)s.state[k]=s.state[k+397]^mpas_mt_twist(s.state[k],s.state[k+1]);
    for(int k=227;k<623;++k)s.state[k]=s.state[k-227]^mpas_mt_twist(s.state[k],s.state[k+1]);
    s.state[623]=s.state[396]^mpas_mt_twist(s.state[623],s.state[0]);
    s.current=0;
}
// MPAS: module_ra_rrtmg_lw.F:1788-1822 (temper, initialize_scalar).
__device__ unsigned int mpas_mt_temper(unsigned int x) {
    x^=x>>11;x^=(x<<7)&0x9d2c5680u;x^=(x<<15)&0xefc60000u;
    return x^(x>>18);
}
// MPAS: module_ra_rrtmg_lw.F:1804-1820.
__device__ void mpas_mt_scalar(mpas_mt_sequence &s,unsigned int seed) {
    s.state[0]=seed;
    for(unsigned int i=1;i<624;++i)s.state[i]=1812433253u*(s.state[i-1]^(s.state[i-1]>>30))+i;
    s.current=624;
}
// MPAS: module_ra_rrtmg_lw.F:1825-1877 (initialize_vector).
__device__ void mpas_mt_vector(mpas_mt_sequence &s,const int *seed,int count) {
    mpas_mt_scalar(s,19650218u);
    int wraps=0,first=max(624,count);
    for(int k=1;k<=first;++k) {
        int i=(k+wraps)%624,j=(k-1)%count;
        if(i==0) {
            s.state[0]=s.state[623];
            s.state[1]=(s.state[1]^((s.state[0]^(s.state[0]>>30))*1664525u))+(unsigned int)seed[j]+j;
            ++wraps;
        } else {
            s.state[i]=(s.state[i]^((s.state[i-1]^(s.state[i-1]>>30))*1664525u))+(unsigned int)seed[j]+j;
        }
    }
    for(int i=first%624+wraps+1;i<624;++i)
        s.state[i]=(s.state[i]^((s.state[i-1]^(s.state[i-1]>>30))*1566083941u))-i;
    s.state[0]=s.state[623];
    for(int i=1;i<=first%624+wraps;++i)
        s.state[i]=(s.state[i]^((s.state[i-1]^(s.state[i-1]>>30))*1566083941u))-i;
    s.state[0]=0x80000000u;s.current=624;
}
// MPAS: module_ra_rrtmg_lw.F:1882-1947 (all three draw operations).
__device__ unsigned int mpas_mt_int(mpas_mt_sequence &s) {
    if(s.current>=624)mpas_mt_next(s);
    return mpas_mt_temper(s.state[s.current++]);
}
// MPAS: module_ra_rrtmg_lw.F:1895-1908.
__device__ unsigned int mpas_mt_positive(mpas_mt_sequence &s) {
    return mpas_mt_int(s)>>1;
}
// MPAS: module_ra_rrtmg_lw.F:1911-1930.
__device__ float mpas_mt_real(mpas_mt_sequence &s) {
    int value=(int)mpas_mt_int(s);
    float result=__int2float_rn(value);
    if(value<0)result=RLW_AD(result,0x1p32f);
    return RLW_DV(result,0x1p32f);
}
// MPAS: module_ra_rrtmg_lw.F:1950-1955 (finalize_RandomNumberSequence).
__device__ void mpas_mt_finalize(mpas_mt_sequence &s) {
    s.current=624;for(int i=0;i<624;++i)s.state[i]=0u;
}
// MPAS: module_ra_rrtmg_lw.F:1993-2030 (scalar/vector stream initialization
// and scalar/1D/2D draws). Fortran 2D order is contiguous first dimension.
extern "C" __global__ void mpas_mt_replay(const int *seed,int count,int vector_seed,
    int stream_only,int n,int *integers,int *positive,float *real,int *state,int *finalized) {
    if(blockIdx.x||threadIdx.x)return;
    mpas_mt_sequence s;
    if(vector_seed)mpas_mt_vector(s,seed,count);else mpas_mt_scalar(s,(unsigned int)seed[0]);
    if(!stream_only) {
        for(int i=0;i<n;++i)integers[i]=(int)mpas_mt_int(s);
        for(int i=0;i<n;++i)positive[i]=(int)mpas_mt_positive(s);
    }
    for(int i=0;i<n;++i)real[i]=mpas_mt_real(s);
    state[0]=s.current;
    for(int i=0;i<624;++i)state[i+1]=(int)s.state[i];
    mpas_mt_finalize(s);finalized[0]=s.current;
    for(int i=0;i<624;++i)finalized[i+1]=(int)s.state[i];
}
// MPAS: module_ra_rrtmg_lw.F:2466-2475,2491-2500,2525-2535,2566-2577,2618-2629.
// MT uses one stream for the entire call, ordered g-point, column, layer.
extern "C" __global__ void mpas_mcica_mt_draws(int nc,int nl,int overlap,int seed,float *cdf,float *cdf2) {
    if(blockIdx.x||threadIdx.x)return;
    mpas_mt_sequence mt;mpas_mt_scalar(mt,(unsigned int)seed);
    for(int g=0;g<140;++g)for(int c=0;c<nc;++c) {
        float shared=overlap==3?mpas_mt_real(mt):0.0f;
        for(int k=0;k<nl;++k) {
            int index=(g*nc+c)*nl+k;
            cdf[index]=overlap==3?shared:mpas_mt_real(mt);
            if(overlap>=4)cdf2[index]=mpas_mt_real(mt);
        }
    }
}
"""


def _lw_dev_chunk(cp, array, rows, cols=None):
    """Device movement for MPAS: module_ra_rrtmg_lw.F:11197-11533."""
    if not isinstance(array, cp.ndarray) or array.dtype != cp.float32:
        raise TypeError('MPAS radiation solver requires device float32 arrays')
    sub = array[rows] if cols is None else array[(rows,) + cols]
    return cp.ascontiguousarray(sub)


def _lw_dev_mcica(cp, array, c0, c1, nl, layout):
    """Device layout of MPAS: module_ra_rrtmg_lw.F:2099-2248."""
    if layout == 'column':
        return _lw_dev_chunk(cp, array, slice(c0,c1),
                             (slice(None,nl),slice(None)))
    if not isinstance(array, cp.ndarray) or array.dtype != cp.float32:
        raise TypeError('MPAS McICA requires device float32 arrays')
    return cp.ascontiguousarray(array[:,c0:c1,:nl].transpose(1,2,0))


# Routine anchors for the complete MPAS module; audited reuse and new code.
# MPAS source window hashes: receipts/source-routines-v842.json.
_MPAS_ROUTINE_CITATIONS = {
    'mixbits': 'MPAS: module_ra_rrtmg_lw.F:1751-1756',
    'twist': 'MPAS: module_ra_rrtmg_lw.F:1758-1767',
    'nextstate': 'MPAS: module_ra_rrtmg_lw.F:1769-1787',
    'temper': 'MPAS: module_ra_rrtmg_lw.F:1789-1800',
    'initialize_scalar': 'MPAS: module_ra_rrtmg_lw.F:1804-1820',
    'initialize_vector': 'MPAS: module_ra_rrtmg_lw.F:1822-1874',
    'getrandomint': 'MPAS: module_ra_rrtmg_lw.F:1878-1893',
    'getrandompositiveint': 'MPAS: module_ra_rrtmg_lw.F:1895-1908',
    'getrandomreal': 'MPAS: module_ra_rrtmg_lw.F:1911-1930',
    'finalize_randomnumbersequence': 'MPAS: module_ra_rrtmg_lw.F:1932-1937',
    'initializerandomnumberstream_s': 'MPAS: module_ra_rrtmg_lw.F:1983-1989',
    'initializerandomnumberstream_v': 'MPAS: module_ra_rrtmg_lw.F:1991-1997',
    'getrandomnumber_scalar': 'MPAS: module_ra_rrtmg_lw.F:2001-2006',
    'getrandomnumber_1d': 'MPAS: module_ra_rrtmg_lw.F:2008-2018',
    'getrandomnumber_2d': 'MPAS: module_ra_rrtmg_lw.F:2020-2030',
    'mcica_subcol_lw': 'MPAS: module_ra_rrtmg_lw.F:2099-2248',
    'generate_stochastic_clouds': 'MPAS: module_ra_rrtmg_lw.F:2252-2697',
    'kissvec': 'MPAS: module_ra_rrtmg_lw.F:2705-2737',
    'cldprmc': 'MPAS: module_ra_rrtmg_lw.F:2775-3036',
    'rtrnmc': 'MPAS: module_ra_rrtmg_lw.F:3096-3533',
    'setcoef': 'MPAS: module_ra_rrtmg_lw.F:3567-3932',
    'lwatmref': 'MPAS: module_ra_rrtmg_lw.F:3935-4095',
    'lwavplank': 'MPAS: module_ra_rrtmg_lw.F:4098-4801',
    'taugb1': 'MPAS: module_ra_rrtmg_lw.F:5084-5177',
    'taugb2': 'MPAS: module_ra_rrtmg_lw.F:5180-5249',
    'taugb3': 'MPAS: module_ra_rrtmg_lw.F:5252-5564',
    'taugb4': 'MPAS: module_ra_rrtmg_lw.F:5567-5823',
    'taugb5': 'MPAS: module_ra_rrtmg_lw.F:5826-6098',
    'taugb6': 'MPAS: module_ra_rrtmg_lw.F:6101-6186',
    'taugb7': 'MPAS: module_ra_rrtmg_lw.F:6189-6460',
    'taugb8': 'MPAS: module_ra_rrtmg_lw.F:6463-6583',
    'taugb9': 'MPAS: module_ra_rrtmg_lw.F:6586-6846',
    'taugb10': 'MPAS: module_ra_rrtmg_lw.F:6849-6913',
    'taugb11': 'MPAS: module_ra_rrtmg_lw.F:6916-6993',
    'taugb12': 'MPAS: module_ra_rrtmg_lw.F:6996-7197',
    'taugb13': 'MPAS: module_ra_rrtmg_lw.F:7200-7456',
    'taugb14': 'MPAS: module_ra_rrtmg_lw.F:7459-7517',
    'taugb15': 'MPAS: module_ra_rrtmg_lw.F:7520-7742',
    'taugb16': 'MPAS: module_ra_rrtmg_lw.F:7745-7951',
    'taumol': 'MPAS: module_ra_rrtmg_lw.F:4835-7953',
    'rrtmg_lw_ini': 'MPAS: module_ra_rrtmg_lw.F:7987-8134',
    'lwdatinit': 'MPAS: module_ra_rrtmg_lw.F:8137-8246',
    'lwcmbdat': 'MPAS: module_ra_rrtmg_lw.F:8249-8328',
    'cmbgb1': 'MPAS: module_ra_rrtmg_lw.F:8331-8440',
    'cmbgb2': 'MPAS: module_ra_rrtmg_lw.F:8443-8523',
    'cmbgb3': 'MPAS: module_ra_rrtmg_lw.F:8526-8653',
    'cmbgb4': 'MPAS: module_ra_rrtmg_lw.F:8656-8752',
    'cmbgb5': 'MPAS: module_ra_rrtmg_lw.F:8755-8878',
    'cmbgb6': 'MPAS: module_ra_rrtmg_lw.F:8881-8969',
    'cmbgb7': 'MPAS: module_ra_rrtmg_lw.F:8972-9093',
    'cmbgb8': 'MPAS: module_ra_rrtmg_lw.F:9096-9212',
    'cmbgb9': 'MPAS: module_ra_rrtmg_lw.F:9215-9337',
    'cmbgb10': 'MPAS: module_ra_rrtmg_lw.F:9340-9424',
    'cmbgb11': 'MPAS: module_ra_rrtmg_lw.F:9427-9527',
    'cmbgb12': 'MPAS: module_ra_rrtmg_lw.F:9530-9599',
    'cmbgb13': 'MPAS: module_ra_rrtmg_lw.F:9602-9712',
    'cmbgb14': 'MPAS: module_ra_rrtmg_lw.F:9715-9799',
    'cmbgb15': 'MPAS: module_ra_rrtmg_lw.F:9802-9886',
    'cmbgb16': 'MPAS: module_ra_rrtmg_lw.F:9889-9982',
    'lwcldpr': 'MPAS: module_ra_rrtmg_lw.F:9985-10623',
    'rrtmg_lw': 'MPAS: module_ra_rrtmg_lw.F:10705-11194',
    'inatm': 'MPAS: module_ra_rrtmg_lw.F:11197-11533',
    'rrtmg_lwrad': 'MPAS: module_ra_rrtmg_lw.F:11574-12305',
    'inirad': 'MPAS: module_ra_rrtmg_lw.F:12308-12331',
    'o3data': 'MPAS: module_ra_rrtmg_lw.F:12334-12444',
    'relcalc': 'MPAS: module_ra_rrtmg_lw.F:12447-12506',
    'reicalc': 'MPAS: module_ra_rrtmg_lw.F:12508-12535',
}


def _solver_source():
    """MPAS: module_ra_rrtmg_lw.F:2775-11533; shared device libm.

    The audited arithmetic is reused. Replace its private legacy libm
    block with the pinned engine's shared glibc header. Host-only
    libm_identity.py/rkind_libm.py do not expose CUDA headers at this base.
    """
    from pathlib import Path
    source = _legacy._gpu_source()
    first = source.index('__device__ const double RLW_LOGF_INVC')
    last = source.index('__device__ float rlw_pow4')
    shared = (Path(_legacy.__file__).parent / 'kernels' /
              'glibc_flt32.cuh').read_text()
    aliases = '\n#define rlw_log gfk_log\n#define rlw_exp gfk_exp\n#define rlw_pow gfk_pow\n'
    anchors='\n'.join('// '+name+': '+cite for name,cite in _MPAS_ROUTINE_CITATIONS.items())+'\n'
    return anchors + source[:first] + shared + aliases + source[last:]

# MPAS: module_ra_rrtmg_lw.F:11538-11747 static wrapper tables.
_DRIVER_SOURCE = r"""
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_43B68000_DEFINED
#define HEX_DIVISOR_43B68000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_43b68000 = 365.0f;
#else
static const float hex_divisor_43b68000 = 365.0f;
#endif
#endif

__device__ const float mpas_retab[95] = {5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};
__device__ const float mpas_pprof[60] = {1000.00f,855.47f,731.82f,626.05f,535.57f,458.16f,391.94f,335.29f,286.83f,245.38f,209.91f,179.57f,153.62f,131.41f,112.42f,96.17f,82.27f,70.38f,60.21f,51.51f,44.06f,37.69f,32.25f,27.59f,23.60f,20.19f,17.27f,14.77f,12.64f,10.81f,9.25f,7.91f,6.77f,5.79f,4.95f,4.24f,3.63f,3.10f,2.65f,2.27f,1.94f,1.66f,1.42f,1.22f,1.04f,0.89f,0.76f,0.65f,0.56f,0.48f,0.41f,0.35f,0.30f,0.26f,0.22f,0.19f,0.16f,0.14f,0.12f,0.10f};
__device__ const float mpas_tprof[60] = {286.96f,281.07f,275.16f,268.11f,260.56f,253.02f,245.62f,238.41f,231.57f,225.91f,221.72f,217.79f,215.06f,212.74f,210.25f,210.16f,210.69f,212.14f,213.74f,215.37f,216.82f,217.94f,219.03f,220.18f,221.37f,222.64f,224.16f,225.88f,227.63f,229.51f,231.50f,233.73f,236.18f,238.78f,241.60f,244.44f,247.35f,250.33f,253.32f,256.30f,259.22f,262.12f,264.80f,266.50f,267.59f,268.44f,268.69f,267.76f,266.13f,263.96f,261.54f,258.93f,256.15f,253.23f,249.89f,246.67f,243.48f,240.25f,236.66f,233.86f};
__device__ const float mpas_o3sum[31] = {5.297E-8f,5.852E-8f,6.579E-8f,7.505E-8f,8.577E-8f,9.895E-8f,1.175E-7f,1.399E-7f,1.677E-7f,2.003E-7f,2.571E-7f,3.325E-7f,4.438E-7f,6.255E-7f,8.168E-7f,1.036E-6f,1.366E-6f,1.855E-6f,2.514E-6f,3.240E-6f,4.033E-6f,4.854E-6f,5.517E-6f,6.089E-6f,6.689E-6f,1.106E-5f,1.462E-5f,1.321E-5f,9.856E-6f,5.960E-6f,5.960E-6f};
__device__ const float mpas_ppsum[31] = {955.890f,850.532f,754.599f,667.742f,589.841f,519.421f,455.480f,398.085f,347.171f,301.735f,261.310f,225.360f,193.419f,165.490f,141.032f,120.125f,102.689f,87.829f,75.123f,64.306f,55.086f,47.209f,40.535f,34.795f,29.865f,19.122f,9.277f,4.660f,2.421f,1.294f,0.647f};
__device__ const float mpas_o3win[31] = {4.629E-8f,4.686E-8f,5.017E-8f,5.613E-8f,6.871E-8f,8.751E-8f,1.138E-7f,1.516E-7f,2.161E-7f,3.264E-7f,4.968E-7f,7.338E-7f,1.017E-6f,1.308E-6f,1.625E-6f,2.011E-6f,2.516E-6f,3.130E-6f,3.840E-6f,4.703E-6f,5.486E-6f,6.289E-6f,6.993E-6f,7.494E-6f,8.197E-6f,9.632E-6f,1.113E-5f,1.146E-5f,9.389E-6f,6.135E-6f,6.135E-6f};
__device__ const float mpas_ppwin[31] = {955.747f,841.783f,740.199f,649.538f,568.404f,495.815f,431.069f,373.464f,322.354f,277.190f,237.635f,203.433f,174.070f,148.949f,127.408f,108.915f,93.114f,79.551f,67.940f,58.072f,49.593f,42.318f,36.138f,30.907f,26.362f,16.423f,7.583f,3.620f,1.807f,0.938f,0.469f};
// MPAS: module_ra_rrtmg_lw.F:11574-12121. One model column, bottom to top.
// m fields: pressure,temperature,dz,qv,qc,qi,qs,cloudfraction,re_cloud,
// re_ice,re_snow,exner. s:skin_temperature,emissivity,xland,xice,snow,lat.
#define M(f,k) m[(f)*nm+(k)]
#define S(f) surface[(f)]
#define A(f,k) a[(f)*nl+(k)]
#define P(k) lev[(k)]
#define T(k) lev[nl+1+(k)]
// MPAS: module_ra_rrtmg_lw.F:11954-11983.
__device__ float mpas_profile(float p) {
    // MPAS: module_ra_rrtmg_lw.F:11954-11983.
    int j=59;
    if (mpas_pprof[59]<p) {
        for(int kk=1;kk<60;++kk) if(mpas_pprof[kk]<p){j=kk-1;break;}
    }
    float weight=j==59?0.0f:RLW_DV(RLW_SU(p,mpas_pprof[j]),RLW_SU(mpas_pprof[j+1],mpas_pprof[j]));
    float next=mpas_tprof[j==59?j:j+1];
    return RLW_AD(RLW_MU(weight,RLW_SU(next,mpas_tprof[j])),mpas_tprof[j]);
}
// MPAS: module_ra_rrtmg_lw.F:12050-12071,12508-12535.
__device__ float mpas_ice_radius(float t,int cap){
    // MPAS: module_ra_rrtmg_lw.F:12050-12071,12508-12535.
    int j=min(max((int)RLW_SU(t,179.0f),1),cap)-1;
    float corr=RLW_SU(t,(float)(int)t);
    return RLW_AD(RLW_MU(mpas_retab[j],RLW_SU(1.0f,corr)),RLW_MU(mpas_retab[j+1],corr));
}
// MPAS: module_ra_rrtmg_lw.F:12447-12506.
__device__ float mpas_liquid_radius(float t,float land,float snow,float ice){
    // MPAS: module_ra_rrtmg_lw.F:12447-12506.
    float r=RLW_AD(8.0f,RLW_MU(6.0f,fminf(1.0f,fmaxf(0.0f,RLW_MU(RLW_SU(273.16f,t),0.05f)))));
    r=RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),fminf(1.0f,fmaxf(0.0f,RLW_MU(RLW_MU(0.001f,snow),10.0f)))));
    r=RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),fminf(1.0f,fmaxf(0.0f,RLW_SU(1.0f,RLW_SU(2.0f,land))))));
    return RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),fminf(1.0f,fmaxf(0.0f,ice))));
}
// MPAS: module_ra_rrtmg_vinterp.F:21-179.
__device__ float mpas_vinterp_one(float p,const float* pin,const float* ozone,int nz,int* upper){
    // MPAS: module_ra_rrtmg_vinterp.F:21-179. Called top to bottom.
    for(int kk=*upper;kk<nz-1;++kk) {
        if(pin[kk]<p && p<=pin[kk+1]){*upper=kk;break;}
    }
    if(p<pin[0]) return RLW_DV(RLW_MU(ozone[0],p),pin[0]);
    if(p>pin[nz-1]) return ozone[nz-1];
    // MPAS: module_ra_rrtmg_vinterp.F:155-161. The one-level native
    // boundary continues with stable complete output words across both
    // hosts, repeats and process counts 1/2. Bound its missing lower words
    // with the zero continuation qualified against those pristine runs.
    float lower_pin=(*upper+1<nz)?pin[*upper+1]:0.0f;
    float lower_ozone=(*upper+1<nz)?ozone[*upper+1]:0.0f;
    float dpu=RLW_SU(p,pin[*upper]),dpl=RLW_SU(lower_pin,p);
    return RLW_DV(RLW_AD(RLW_MU(ozone[*upper],dpl),RLW_MU(lower_ozone,dpu)),RLW_AD(dpl,dpu));
}
// MPAS: module_ra_rrtmg_vinterp.F:114-120.
extern "C" __global__ void mpas_vinterp_start(int nc,int nz,const int* upper,int* start){
    // MPAS: module_ra_rrtmg_vinterp.F:114-120. One minimum across all columns.
    if(blockIdx.x || threadIdx.x)return;
    int kk=nz-1;for(int c=0;c<nc;++c)kk=min(kk,upper[c]);*start=kk;
}
// MPAS: module_ra_rrtmg_vinterp.F:122-165.
extern "C" __global__ void mpas_vinterp_level(int nc,int nl,int nz,int k,const float* pmid,
    const float* pin,const float* ozone,const int* start,int* upper,float* out){
    // MPAS: module_ra_rrtmg_vinterp.F:122-165; common search start at each level.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    float p=pmid[c*nl+k];int u=upper[c];
    for(int kk=*start;kk<nz-1;++kk)if(pin[kk]<p && p<=pin[kk+1]){u=kk;break;}
    upper[c]=u;
    if(p<pin[0])out[c*nl+k]=RLW_DV(RLW_MU(ozone[c*nz],p),pin[0]);
    else if(p>pin[nz-1])out[c*nl+k]=ozone[c*nz+nz-1];
    else{
        // MPAS: module_ra_rrtmg_vinterp.F:155-161. Same measured
        // one-level continuation as the complete LW wrapper helper.
        float lower_pin=(u+1<nz)?pin[u+1]:0.0f;
        float lower_ozone=(u+1<nz)?ozone[c*nz+u+1]:0.0f;
        float dpu=RLW_SU(p,pin[u]),dpl=RLW_SU(lower_pin,p);
        out[c*nl+k]=RLW_DV(RLW_AD(RLW_MU(ozone[c*nz+u],dpl),
            RLW_MU(lower_ozone,dpu)),RLW_AD(dpl,dpu));
    }
}
// MPAS: module_ra_rrtmg_vinterp.F:114-165. mpas_vinterp_level with the next
// level's common search start reduced on the device: starts[k] holds
// min(nz-1, every column's upper after level k+1), as mpas_vinterp_start
// computes it serially. THE BREAKAGE THIS PREVENTS: the one-thread start
// kernel walked all 216,187 columns at every level of every LW call (G2).
extern "C" __global__ void mpas_vinterp_level_chain(int nc,int nl,int nz,int k,const float* pmid,
    const float* pin,const float* ozone,int* starts,int* upper,float* out){
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    int u=0x7fffffff;
    if(c<nc){
        float p=pmid[c*nl+k];u=upper[c];
        for(int kk=starts[k];kk<nz-1;++kk)if(pin[kk]<p && p<=pin[kk+1]){u=kk;break;}
        upper[c]=u;
        if(p<pin[0])out[c*nl+k]=RLW_DV(RLW_MU(ozone[c*nz],p),pin[0]);
        else if(p>pin[nz-1])out[c*nl+k]=ozone[c*nz+nz-1];
        else{
            float lower_pin=(u+1<nz)?pin[u+1]:0.0f;
            float lower_ozone=(u+1<nz)?ozone[c*nz+u+1]:0.0f;
            float dpu=RLW_SU(p,pin[u]),dpl=RLW_SU(lower_pin,p);
            out[c*nl+k]=RLW_DV(RLW_AD(RLW_MU(ozone[c*nz+u],dpl),
                RLW_MU(lower_ozone,dpu)),RLW_AD(dpl,dpu));
        }
    }
    if(k==0)return;
    for(int offset=16;offset>0;offset>>=1)u=min(u,__shfl_down_sync(0xffffffffu,u,offset));
    if((threadIdx.x&31)==0)atomicMin(starts+k-1,u);
}
// MPAS: module_ra_rrtmg_lw.F:11817-11821.
extern "C" __global__ void mpas_layer_counts(int nc,int nm,const float* interfaces,int* counts){
    // MPAS: module_ra_rrtmg_lw.F:11817-11821; NINT ties away from zero.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    float top=RLW_DV(interfaces[c*2*(nm+1)+nm],100.0f);
    counts[c]=nm+max((int)RLW_AD(RLW_DV(top,4.0f),0.5f),1);
}
// MPAS: module_ra_rrtmg_lw.F:11810-12164.
__device__ void mpas_lw_prepare_one(int nm,int nl,int icloud,int reqc,int reqi,int reqs,int o3input,int nz,int presence,
 const float* m,const float* interfaces,const float* surface,const float* pin,const float* ozone,float* a,float* lev){
    float dzsum=0.0f;
    for(int k=0;k<=nm;++k){P(k)=RLW_DV(interfaces[k],100.0f);T(k)=interfaces[nm+1+k];}
    for(int k=0;k<nl;++k){
        float dz=M(2,min(k,nm-1));
        A(2,k)=RLW_AD(dzsum,RLW_MU(0.5f,dz));dzsum=RLW_AD(dzsum,dz);
        A(3,k)=RLW_MU(fmaxf(M(3,min(k,nm-1)),1.e-12f),1.607793f);
        if(k<nm){A(0,k)=RLW_DV(M(0,k),100.0f);A(1,k)=M(1,k);}
        else {P(k+1)=fmaxf(RLW_SU(P(k),4.0f),0.0f);A(0,k)=RLW_MU(0.5f,RLW_AD(P(k),P(k+1)));}
    }
    P(nl)=0.0f;A(0,nl-1)=RLW_MU(0.5f,RLW_AD(P(nl-1),P(nl)));
    float anchor=RLW_SU(T(nm-1),mpas_profile(P(nm-1)));
    for(int k=nm;k<=nl;++k){T(k)=RLW_AD(mpas_profile(P(k)),anchor);A(1,k-1)=RLW_MU(0.5f,RLW_AD(T(k),T(k-1)));}
    // MPAS: module_ra_rrtmg_lw.F:12308-12444 (INIRAD/O3DATA).
    float annual[31],phalf[32];phalf[0]=1100.0f;phalf[31]=0.0f;
    annual[0]=RLW_MU(0.5f,RLW_AD(mpas_o3sum[0],mpas_o3win[0]));
    for(int j=1;j<31;++j){
        float interp=RLW_AD(mpas_o3win[j-1],RLW_MU(RLW_DV(RLW_SU(mpas_o3win[j],mpas_o3win[j-1]),RLW_SU(mpas_ppwin[j],mpas_ppwin[j-1])),RLW_SU(mpas_ppsum[j],mpas_ppwin[j-1])));
        annual[j]=RLW_MU(0.5f,RLW_AD(interp,mpas_o3sum[j]));
        phalf[j]=RLW_DV(RLW_AD(mpas_ppsum[j],mpas_ppsum[j-1]),2.0f);
    }
    for(int k=0;k<nl;++k){
        float o3=0.0f;
        for(int j=0;j<31;++j){
            float pb1=fmaxf(RLW_SU(P(k),phalf[j]),0.0f),pb2=fmaxf(RLW_SU(P(k),phalf[j+1]),0.0f);
            float pt1=fmaxf(RLW_SU(P(k+1),phalf[j]),0.0f),pt2=fmaxf(RLW_SU(P(k+1),phalf[j+1]),0.0f);
            o3=RLW_AD(o3,RLW_MU(RLW_AD(RLW_SU(RLW_SU(pb2,pb1),pt2),pt1),annual[j]));
        }
        A(4,k)=RLW_MU(RLW_DV(o3,RLW_SU(P(k),P(k+1))),0.603461f);
    }
    if(o3input==2){int upper=0;for(int k=nl-1;k>=0;--k) A(4,k)=mpas_vinterp_one(A(0,k),pin,ozone,nz,&upper);}
    // MPAS: module_ra_rrtmg_lw.F:12016-12164.
    for(int k=0;k<nl;++k){
        // MPAS: module_ra_rrtmg_lw.F:11801-11806. PRESENT is independent of mass.
        float cf=k<nm&&icloud>0&&(presence&7)&&(presence&8)?M(7,k):0.0f;A(5,k)=cf;
        float rel=10.0f,rei=10.0f,res=10.0f,liq=0.0f,ice=0.0f,snow=0.0f;
        if(k<nm){
            if(reqc){rel=fmaxf(2.5f,RLW_MU(M(8,k),1.e6f));if(rel<=2.5f&&cf>0.0f){if(S(2)>1.5f)rel=10.5f;else if(S(2)<1.5f)rel=7.5f;}}
            else rel=mpas_liquid_radius(A(1,k),S(2),S(4),S(3));
            if(reqi){rei=fmaxf(5.0f,RLW_MU(M(9,k),1.e6f));if(rei<=5.0f&&cf>0.0f)rei=fmaxf(5.0f,mpas_ice_radius(M(1,k),75));}
            else{rei=mpas_ice_radius(A(1,k),94);rei=fminf(140.0f,RLW_MU(rei,1.0315f));}
            if(reqs)res=fmaxf(10.0f,RLW_MU(M(10,k),1.e6f));
            float dp=RLW_SU(P(k),P(k+1)),denom=fmaxf(0.01f,cf);
            float qc=icloud>0&&(presence&1)?fmaxf(M(4,k),0.0f):0.0f;
            float qi=icloud>0&&(presence&2)?fmaxf(M(5,k),0.0f):0.0f;
            float qs=icloud>0&&(presence&4)?fmaxf(M(6,k),0.0f):0.0f;
            liq=RLW_DV(RLW_MU(RLW_DV(RLW_MU(RLW_MU(qc,dp),100.0f),9.80616f),1000.0f),denom);
            float qice=reqi||reqs?qi:RLW_AD(qi,qs);
            ice=RLW_DV(RLW_MU(RLW_DV(RLW_MU(RLW_MU(qice,dp),100.0f),9.80616f),1000.0f),denom);
            if(reqs){float factor=1.0f;if(res>130.0f){float ratio=RLW_DV(130.0f,res);factor=RLW_MU(ratio,ratio);res=130.0f;}
                snow=RLW_DV(RLW_MU(factor,RLW_MU(RLW_DV(RLW_MU(RLW_MU(qs,dp),100.0f),9.80616f),1000.0f)),denom);}
        }
        A(6,k)=liq;A(7,k)=ice;A(8,k)=snow;A(9,k)=rel;A(10,k)=rei;A(11,k)=res;
    }
}
// MPAS: module_ra_rrtmg_lw.F:11810-12164.
extern "C" __global__ void mpas_lw_prepare(int nc,int nm,int nl,int icloud,int reqc,int reqi,int reqs,int o3input,int nz,int presence,
 const float* m,const float* interfaces,const float* surface,const float* pin,const float* ozone,float* a,float* lev){
    // MPAS: module_ra_rrtmg_lw.F:11810-12164, independent column loop.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    mpas_lw_prepare_one(nm,nl,icloud,reqc,reqi,reqs,o3input,nz,presence,m+c*12*nm,
        interfaces+c*2*(nm+1),surface+c*6,pin,ozone+(o3input==2?c*nz:0),a+c*12*nl,lev+c*2*(nl+1));
}
// MPAS: module_ra_rrtmg_lw.F:2705-2737.
__device__ float mpas_kiss(unsigned int* seed){
    // MPAS: module_ra_rrtmg_lw.F:2705-2737. Unsigned implements wrap/ISHFT.
    seed[0]=69069u*seed[0]+1327217885u;
    seed[1]^=seed[1]<<13;seed[1]^=seed[1]>>17;seed[1]^=seed[1]<<5;
    seed[2]=18000u*(seed[2]&65535u)+(seed[2]>>16);
    seed[3]=30903u*(seed[3]&65535u)+(seed[3]>>16);
    int kiss=(int)(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);
    return RLW_AD(RLW_MU((float)kiss,2.328306e-10f),0.5f);
}
// MPAS: module_ra_rrtmg_lw.F:2252-2697.
__device__ void mpas_mcica_make(int nl,int overlap,int idcor,int julday,const float* surface,const float* a,
    int irng,int permuteseed,const float* cdf_input,const float* cdf2_input,int cdf_stride,
    const float* tauc,const int* ngb,float* tau,
    float* cloud,float* liquid,float* ice,float* snow){
    // MPAS: module_ra_rrtmg_lw.F:2099-2737. One MPAS wrapper column.
    if(overlap==0)return;
    unsigned int seed[4];mpas_mt_sequence mt;
    if(irng==0){
        // MPAS: module_ra_rrtmg_lw.F:2432-2440. Short-column native OOB seed
        // words measured as zero in the pristine McICA boundary matrix; full
        // wrapper short-column run-to-run variability remains unqualified.
        for(int k=0;k<4;++k){seed[k]=0u;if(k<nl){float p=RLW_MU(A(0,k),100.0f);seed[k]=(unsigned int)(int)RLW_MU(RLW_SU(p,(float)(int)p),1.e9f);}}
        for(int j=0;j<permuteseed;++j)mpas_kiss(seed);
    }else if(!cdf_input)mpas_mt_scalar(mt,(unsigned int)permuteseed);
    float length=2500.0f;
    if(idcor==1){
        float am3=RLW_MU(julday>181?(-28.0f/hex_divisor_43b68000):(28.0f/hex_divisor_43b68000),(float)(julday-(julday>181?272:91)));
        float delta=RLW_SU(surface[5],am3);
        length=RLW_MU(RLW_AD(1.4315f,RLW_MU(2.1219f,rlw_exp(RLW_DV(-RLW_MU(delta,delta),(-25.584f*-25.584f))))),1000.0f);
    }
    float inverse=RLW_DV(1.0f,length);
    for(int ig=0;ig<140;++ig){
        float prev=0.0f,prevcf=0.0f,shared=0.0f;
        if(overlap==3)shared=cdf_input?cdf_input[ig*cdf_stride]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt));
        for(int k=0;k<nl;++k){
            float cf=A(5,k);if(cf<1.e-20f)cf=0.0f;
            float cdf=overlap==3?shared:(cdf_input?cdf_input[ig*cdf_stride+k]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt)));
            if(overlap>=4){float cdf2=cdf2_input?cdf2_input[ig*cdf_stride+k]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt));
                float alpha=k?rlw_exp(RLW_MU(-RLW_SU(A(2,k),A(2,k-1)),inverse)):0.0f;
                if(overlap==5&&cf==0.0f&&prevcf>0.0f)alpha=0.0f;
                if(k&&cdf2<alpha)cdf=prev;
            }else if(overlap==2&&k){float threshold=RLW_SU(1.0f,prevcf);cdf=prev>threshold?prev:RLW_MU(cdf,threshold);}
            bool cloudy=cdf>=RLW_SU(1.0f,cf);int idx=k*140+ig;
            cloud[idx]=cloudy?1.0f:0.0f;liquid[idx]=cloudy?A(6,k):0.0f;ice[idx]=cloudy?A(7,k):0.0f;snow[idx]=cloudy?A(8,k):0.0f;
            if(tau)tau[idx]=cloudy?tauc[(ngb[ig]-1)*nl+k]:0.0f;
            prev=cdf;prevcf=cf;
        }
    }
}
// MPAS: module_ra_rrtmg_lw.F:12175-12185.
extern "C" __global__ void mpas_lw_mcica(int nc,int nl,int overlap,int idcor,int julday,const float* surface,const float* a,float* cloud,float* liquid,float* ice,float* snow){
    // MPAS: module_ra_rrtmg_lw.F:12175-12185 fixes IRNG=0, seed=150.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    mpas_mcica_make(nl,overlap,idcor,julday,surface+c*6,a+c*12*nl,0,150,nullptr,nullptr,0,nullptr,nullptr,nullptr,
        cloud+c*nl*140,liquid+c*nl*140,ice+c*nl*140,snow+c*nl*140);
}
// MPAS: module_ra_rrtmg_lw.F:2705-2737. Jump-ahead of the KISS components, so
// each g-point of a column starts from exactly the state the sequential stream
// reaches there. The four components evolve independently: the LCG and the
// xorshift are advanced by their own n-step maps (affine, and linear over
// GF(2) from a host table), and each multiply-with-carry s=a*(s&65535)+(s>>16)
// is, once in its canonical range s <= m = a*65536-1 (reached within three
// steps from any word), multiplication by a modulo m (b*s' = m*x + s, so
// s' = a*s mod m; s = 0 and s = m are fixed points).
__device__ unsigned int mpas_lcg_jump(unsigned int s,unsigned int n){
    unsigned int ra=1u,rc=0u,pa=69069u,pc=1327217885u;
    while(n){if(n&1u){rc=pa*rc+pc;ra=pa*ra;}pc=pa*pc+pc;pa=pa*pa;n>>=1;}
    return ra*s+rc;
}
__device__ unsigned int mpas_mwc_jump(unsigned int s,unsigned int a,unsigned int n){
    const unsigned long long m=(unsigned long long)a*65536ull-1ull;
    while(n&&(unsigned long long)s>m){s=a*(s&65535u)+(s>>16);--n;}
    if(!n||(unsigned long long)s==m||s==0u)return s;
    unsigned long long r=1ull,base=a;
    while(n){if(n&1u)r=r*base%m;base=base*base%m;n>>=1;}
    return (unsigned int)((unsigned long long)s*r%m);
}
__device__ unsigned int mpas_xorshift_jump(unsigned int s,const unsigned int* columns){
    unsigned int v=0u;
    for(int i=0;i<32;++i)if((s>>i)&1u)v^=columns[i];
    return v;
}
// MPAS: module_ra_rrtmg_lw.F:2432-2440,12175-12185 (IRNG=0, seed=150), 2252-2697.
// Per wrapper column: the KISS state after the permutation draws and the
// per-layer exponential overlap weights, with mpas_mcica_make's arithmetic
// (the same rlw_exp of the same operands).
extern "C" __global__ void mpas_lw_mcica_seeds(int nc,int nl,int overlap,int idcor,int julday,int permuteseed,
    const float* surface,const float* abase,unsigned int* seeds,float* weights){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    const float* a=abase+(long long)c*12*nl;
    unsigned int seed[4];
    for(int k=0;k<4;++k){seed[k]=0u;if(k<nl){float p=RLW_MU(A(0,k),100.0f);seed[k]=(unsigned int)(int)RLW_MU(RLW_SU(p,(float)(int)p),1.e9f);}}
    for(int j=0;j<permuteseed;++j)mpas_kiss(seed);
    for(int k=0;k<4;++k)seeds[c*4+k]=seed[k];
    if(overlap<4)return;
    float length=2500.0f;
    if(idcor==1){
        float am3=RLW_MU(julday>181?(-28.0f/hex_divisor_43b68000):(28.0f/hex_divisor_43b68000),(float)(julday-(julday>181?272:91)));
        float delta=RLW_SU(surface[c*6+5],am3);
        length=RLW_MU(RLW_AD(1.4315f,RLW_MU(2.1219f,rlw_exp(RLW_DV(-RLW_MU(delta,delta),(-25.584f*-25.584f))))),1000.0f);
    }
    float inverse=RLW_DV(1.0f,length);
    for(int k=0;k<nl;++k)weights[(long long)c*nl+k]=k?rlw_exp(RLW_MU(-RLW_SU(A(2,k),A(2,k-1)),inverse)):0.0f;
}
// MPAS: module_ra_rrtmg_lw.F:2252-2697, the wrapper's subcolumns, one thread per
// (column, g-point). Each thread jumps its column's stream to its g-point's
// first draw (draws = KISS draws per g-point: 1 maximum, nl random/max-random,
// 2*nl exponential) and runs mpas_mcica_make's layer loop for that g-point
// unchanged. Neighbouring threads hold neighbouring g-points, so the four slabs
// are written as contiguous rows. THE BREAKAGE THIS PREVENTS: one thread per
// column walked 140 x nl draws in sequence and wrote four slabs at a column
// stride, 1.9 s of a 3.5 s radiation call on 216,187 cells (fix group G2).
extern "C" __global__ void mpas_lw_mcica_gpoint(int nc,int nl,int overlap,int draws,const float* abase,
    const unsigned int* seeds,const float* weights,const unsigned int* xorshift,
    float* cloud,float* liquid,float* ice,float* snow){
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)nc*140)return;
    const int ig=(int)(t%140);const long long c=t/140;
    const float* a=abase+c*12*nl;
    const unsigned int n=(unsigned int)ig*(unsigned int)draws;
    unsigned int seed[4];
    seed[0]=mpas_lcg_jump(seeds[c*4],n);
    seed[1]=mpas_xorshift_jump(seeds[c*4+1],xorshift+ig*32);
    seed[2]=mpas_mwc_jump(seeds[c*4+2],18000u,n);
    seed[3]=mpas_mwc_jump(seeds[c*4+3],30903u,n);
    const float* weight=weights+(overlap>=4?c*nl:0);
    float prev=0.0f,prevcf=0.0f,shared=0.0f;
    if(overlap==3)shared=mpas_kiss(seed);
    for(int k=0;k<nl;++k){
        float cf=A(5,k);if(cf<1.e-20f)cf=0.0f;
        float cdf=overlap==3?shared:mpas_kiss(seed);
        if(overlap>=4){float cdf2=mpas_kiss(seed);
            float alpha=weight[k];
            if(overlap==5&&cf==0.0f&&prevcf>0.0f)alpha=0.0f;
            if(k&&cdf2<alpha)cdf=prev;
        }else if(overlap==2&&k){float threshold=RLW_SU(1.0f,prevcf);cdf=prev>threshold?prev:RLW_MU(cdf,threshold);}
        bool cloudy=cdf>=RLW_SU(1.0f,cf);const long long idx=(c*nl+k)*140+ig;
        cloud[idx]=cloudy?1.0f:0.0f;liquid[idx]=cloudy?A(6,k):0.0f;ice[idx]=cloudy?A(7,k):0.0f;snow[idx]=cloudy?A(8,k):0.0f;
        prev=cdf;prevcf=cf;
    }
}
// MPAS: module_ra_rrtmg_lw.F:11817-11821, histogram of the wrapper layer counts.
// Bins [base, base+nbins); anything outside lands in bin nbins (host falls back).
extern "C" __global__ void mpas_layer_histogram(int nc,int base,int nbins,const int* counts,int* hist){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    int b=counts[c]-base;atomicAdd(hist+((b<0||b>=nbins)?nbins:b),1);
}
// MPAS: module_ra_rrtmg_lw.F:2099-2248.
extern "C" __global__ void mpas_mcica_column(int nl,int overlap,int idcor,int julday,const float* surface,const float* a,
    int irng,int permuteseed,const float* tauc,const int* ngb,float* tau,
    float* cloud,float* liquid,float* ice,float* snow,float* radii){
    // MPAS: module_ra_rrtmg_lw.F:2099-2248. Generic one-column interface.
    if(threadIdx.x||blockIdx.x||overlap==0)return;
    for(int k=0;k<nl;++k)for(int j=0;j<3;++j)radii[j*nl+k]=A(9+j,k);
    mpas_mcica_make(nl,overlap,idcor,julday,surface,a,irng,permuteseed,nullptr,nullptr,0,tauc,ngb,tau,cloud,liquid,ice,snow);
}
// MPAS: module_ra_rrtmg_lw.F:2099-2697.
extern "C" __global__ void mpas_mcica_batch(int nc,int nl,int overlap,int idcor,int julday,const float* surface,const float* a,
    int irng,int permuteseed,const float* cdf,const float* cdf2,const float* tauc,const int* ngb,float* tau,
    float* cloud,float* liquid,float* ice,float* snow,float* radii){
    // MPAS: module_ra_rrtmg_lw.F:2099-2697, general multi-column interface.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc||overlap==0)return;
    const float *column=a+c*12*nl;
    for(int k=0;k<nl;++k)for(int j=0;j<3;++j)radii[(c*3+j)*nl+k]=column[(9+j)*nl+k];
    mpas_mcica_make(nl,overlap,idcor,julday,surface+c*6,column,irng,permuteseed,
        irng?cdf+c*nl:nullptr,irng?cdf2+c*nl:nullptr,nc*nl,tauc+c*16*nl,ngb,tau+c*nl*140,
        cloud+c*nl*140,liquid+c*nl*140,ice+c*nl*140,snow+c*nl*140);
}
// MPAS: module_ra_rrtmg_lw.F:12198-12254.
__device__ void mpas_lw_outputs_one(int nm,int nl,int icloud,const float* m,const float* a,const float* u,const float* d,const float* uc,const float* dc,const float* hr,float* tendency,float* radii,float* flux){
    // MPAS: module_ra_rrtmg_lw.F:12198-12254; driver _to_MPAS:613-719.
    flux[0]=d[0];flux[1]=u[nl];flux[2]=RLW_SU(uc[nl],u[nl]);
    flux[3]=u[nl];flux[4]=uc[nl];flux[5]=d[nl];flux[6]=dc[nl];
    flux[7]=u[0];flux[8]=uc[0];flux[9]=d[0];flux[10]=dc[0];
    for(int k=0;k<nm;++k){tendency[k]=RLW_DV(RLW_DV(hr[k],86400.0f),M(11,k));
        for(int j=0;j<3;++j)radii[j*nm+k]=icloud>0&&M(7,k)>0.0f?A(9+j,k):0.0f;}
}
// MPAS: module_ra_rrtmg_lw.F:12198-12254.
extern "C" __global__ void mpas_lw_outputs(int nc,int nm,int nl,int icloud,const float* m,const float* a,const float* u,const float* d,const float* uc,const float* dc,const float* hr,float* tendency,float* radii,float* flux){
    // MPAS: module_ra_rrtmg_lw.F:12198-12254, independent column loop.
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    mpas_lw_outputs_one(nm,nl,icloud,m+c*12*nm,a+c*12*nl,u+c*(nl+1),d+c*(nl+1),
        uc+c*(nl+1),dc+c*(nl+1),hr+c*nl,tendency+c*nm,radii+c*3*nm,flux+c*11);
}
#undef M
#undef S
#undef A
#undef P
#undef T
// MPAS: mpas_atmphys_driver_radiation_lw.F:490-492.
extern "C" __global__ void mpas_pressure_mb(int n,const float *p,float *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=__fdiv_rn(p[i],100.f);
}

"""

def _gpu_kernel(name, nlay=0):
    """MPAS: module_ra_rrtmg_lw.F:2775-11533 plus the driver below."""
    import cupy as cp
    device = int(cp.cuda.Device().id)
    capacity = max(MAX_RADIATION_LAYERS, int(nlay))
    key = (device, capacity)
    if key not in _MODULES:
        from woof import nvrtc_ptx_cache
        source = _solver_source() + "\n" + _RANDOM_SOURCE + "\n" + _DRIVER_SOURCE + "\n" + _INIT_SOURCE
        # MPAS: module_ra_rrtmg_lw.F:3176-3195,3302-3314. Native arrays
        # follow nlay; specialize the reused local arrays to the same size.
        source = source.replace('#define RLW_MAXLAY 128',
                                f'#define RLW_MAXLAY {capacity}')
        ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(
            source, _hex_numerics.options(("-std=c++17", "--ftz=false", "--fmad=false",
                     "--prec-div=true", "--prec-sqrt=true")),
            None, "mpas_rrtmg_lw_v842.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx,str) else ptx)
        _MODULES[key] = module
    return _MODULES[key].get_function(name)

def solve_device(ncol, nlay, icld, play, plev, tlay, tlev,
                                tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr,
                                n2ovmr, o2vmr, cfc11vmr, cfc12vmr,
                                cfc22vmr, ccl4vmr, emis, inflglw,
                                iceflglw, liqflglw, cldfmcl, taucmcl,
                                ciwpmcl, clwpmcl, cswpmcl, reicmcl,
                                relqmcl, resnmcl, tauaer, K,
                                column_chunk=None, _stage_probe=None, *,
                                mcica_layout="gpoint"):
    """MPAS: module_ra_rrtmg_lw.F:10705-11533,2775-10623.

    Audited solver launch chain. K holds device coefficient tables, scalar
    constants and owning references supplied by the radiation table backend.
    """
    if mcica_layout not in ("gpoint", "column"):
        raise ValueError("mcica_layout must be gpoint or column")
    import cupy as cp
    cp.cuda.get_current_stream().wait_event(K['ready'])
    nl = int(nlay)
    ncol = int(ncol)
    chunk = (int(column_chunk) if column_chunk
             else min(ncol,_legacy.batch_column_chunk(
                 NGPTLW,_legacy.LW_BATCH_COLUMN_CHUNK_CEILING)))
    if chunk < 1:
        raise ValueError("column_chunk must be >= 1")

    i32, f32 = np.int32, np.float32

    uflx = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    dflx = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    uflxc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    dflxc = cp.zeros((ncol, nl + 1), dtype=cp.float32)
    hr = cp.zeros((ncol, nl), dtype=cp.float32)
    hrc = cp.zeros((ncol, nl), dtype=cp.float32)

    fslot = {name: i for i, name in enumerate(GPU_FSLOTS)}
    islot = {name: i for i, name in enumerate(GPU_ISLOTS)}

    scratch = LWBatchScratch(cp)
    err_flags = cp.zeros((ncol + chunk - 1) // chunk, dtype=cp.int32)
    for c0 in range(0, ncol, chunk):
        c1 = min(c0 + chunk, ncol)
        nc = c1 - c0
        rows = slice(c0, c1)

        # ---- uploads --------------------------------------------------
        play_d = _lw_dev_chunk(cp, play, rows, (slice(0, nl),))
        plev_d = _lw_dev_chunk(cp, plev, rows, (slice(0, nl + 1),))
        tlay_d = _lw_dev_chunk(cp, tlay, rows, (slice(0, nl),))
        tlev_d = _lw_dev_chunk(cp, tlev, rows, (slice(0, nl + 1),))
        tsfc_d = _lw_dev_chunk(cp, tsfc, rows)
        emis_d = _lw_dev_chunk(cp, emis, rows, (slice(0, NBNDLW),))
        # wkl/wx slot installs are the same pure data movement the
        # per-column gpu_inatm does host-side (unfilled slots stay 0).
        wkl_d = cp.zeros((nc, MXMOL, nl), dtype=cp.float32)
        for slot, arr in ((1, h2ovmr), (2, co2vmr), (3, o3vmr),
                          (4, n2ovmr), (6, ch4vmr), (7, o2vmr)):
            wkl_d[:, slot - 1, :] = _lw_dev_chunk(cp, arr, rows,
                                                  (slice(0, nl),))
        wx_d = cp.zeros((nc, MAXXSEC, nl), dtype=cp.float32)
        for slot, arr in ((1, ccl4vmr), (2, cfc11vmr), (3, cfc12vmr),
                          (4, cfc22vmr)):
            wx_d[:, slot - 1, :] = _lw_dev_chunk(cp, arr, rows,
                                                 (slice(0, nl),))
        cldfmc_d = _lw_dev_mcica(cp, cldfmcl, c0, c1, nl, mcica_layout)
        taucmc_d = _lw_dev_mcica(cp, taucmcl, c0, c1, nl, mcica_layout)
        ciwpmc_d = _lw_dev_mcica(cp, ciwpmcl, c0, c1, nl, mcica_layout)
        clwpmc_d = _lw_dev_mcica(cp, clwpmcl, c0, c1, nl, mcica_layout)
        cswpmc_d = _lw_dev_mcica(cp, cswpmcl, c0, c1, nl, mcica_layout)
        reicmc_d = _lw_dev_chunk(cp, reicmcl, rows, (slice(0, nl),))
        relqmc_d = _lw_dev_chunk(cp, relqmcl, rows, (slice(0, nl),))
        resnmc_d = _lw_dev_chunk(cp, resnmcl, rows, (slice(0, nl),))
        if int(icld) < 1:
            # MPAS: module_ra_rrtmg_lw.F:11364-11375,11500-11524.
            # INATM clears cloud arrays before its conditional input copy.
            cldfmc_d=cp.zeros((nc,nl,NGPTLW),dtype=cp.float32)
            taucmc_d=cp.zeros_like(cldfmc_d)
            ciwpmc_d=cp.zeros_like(cldfmc_d)
            clwpmc_d=cp.zeros_like(cldfmc_d)
            cswpmc_d=cp.zeros_like(cldfmc_d)
            reicmc_d=cp.zeros((nc,nl),dtype=cp.float32)
            relqmc_d=cp.zeros_like(reicmc_d)
            resnmc_d=cp.zeros_like(reicmc_d)
        taua_d = _lw_dev_chunk(cp, tauaer, rows,
                               (slice(0, nl), slice(0, NBNDLW)))

        # ---- inatm derived quantities ----------------------------------
        coldry_d = cp.empty((nc, nl), dtype=cp.float32)
        wbrodl_d = cp.empty((nc, nl), dtype=cp.float32)
        pwvcm_d = cp.empty(nc, dtype=cp.float32)
        _gpu_kernel("rlw_inatm")(
            ((nc + 31) // 32,), (32,),
            (i32(nc), i32(nl), plev_d, K["grav"], K["avogad"],
             wkl_d, wx_d, coldry_d, wbrodl_d, pwvcm_d))

        # ---- cldprmc ----------------------------------------------------
        ncb_flag = cp.zeros(nc, dtype=cp.int32)
        total = nc * NGPTLW
        _gpu_kernel("rlw_cldprmc_coalesced")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), i32(inflglw), i32(iceflglw), i32(liqflglw),
             cldfmc_d, ciwpmc_d, clwpmc_d, cswpmc_d,
             reicmc_d, relqmc_d, resnmc_d,
             K["absice1"], K["absice2"], K["absice3"], K["absice0"],
             K["absliq1"], K["absliq0"], K["ngb"],
             taucmc_d, ncb_flag, err_flags[c0 // chunk:c0 // chunk + 1]))
        if _stage_probe is not None:
            _stage_probe("upload")
        del ciwpmc_d, clwpmc_d, cswpmc_d, reicmc_d, relqmc_d, resnmc_d

        # ---- setcoef, writing straight into the frozen taumol slabs ----
        fs_d = cp.zeros((len(GPU_FSLOTS), nc, nl), dtype=cp.float32)
        isv_d = cp.zeros((len(GPU_ISLOTS), nc, nl), dtype=cp.int32)
        fs_d[fslot["pavel"]] = play_d
        fs_d[fslot["coldry"]] = coldry_d
        laytrop_d = cp.zeros(nc, dtype=cp.int32)
        planklay = cp.empty((nc, nl, NBNDLW), dtype=cp.float32)
        planklev = cp.empty((nc, nl + 1, NBNDLW), dtype=cp.float32)
        plankbnd = cp.empty((nc, NBNDLW), dtype=cp.float32)

        def FV(name):
            """MPAS: module_ra_rrtmg_lw.F:3567-3932."""
            return fs_d[fslot[name]]

        def IV(name):
            """MPAS: module_ra_rrtmg_lw.F:3567-3932."""
            return isv_d[islot[name]]

        _gpu_kernel("rlw_setcoef")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), i32(nl), i32(1),
             FV("pavel"), tlay_d, tlev_d, tsfc_d, emis_d,
             FV("coldry"), wkl_d, wbrodl_d,
             K["totplnk"], K["totplk16"], K["preflog"], K["tref"],
             K["chi"], laytrop_d,
             IV("jp"), IV("jt"), IV("jt1"),
             planklay, planklev, plankbnd,
             FV("colh2o"), FV("colco2"), FV("colo3"), FV("coln2o"),
             FV("colco"), FV("colch4"), FV("colo2"), FV("colbrd"),
             FV("fac00"), FV("fac01"), FV("fac10"), FV("fac11"),
             FV("rat_h2oco2"), FV("rat_h2oco2_1"),
             FV("rat_h2oo3"), FV("rat_h2oo3_1"),
             FV("rat_h2on2o"), FV("rat_h2on2o_1"),
             FV("rat_h2och4"), FV("rat_h2och4_1"),
             FV("rat_n2oco2"), FV("rat_n2oco2_1"),
             FV("rat_o3co2"), FV("rat_o3co2_1"),
             FV("selffac"), FV("selffrac"), IV("indself"),
             FV("forfac"), FV("forfrac"), IV("indfor"),
             FV("minorfrac"), FV("scaleminor"), FV("scaleminorn2"),
             IV("indminor")))
        del play_d, tlay_d, tlev_d, wkl_d, wbrodl_d, coldry_d

        # ---- taumol: the 16 band kernels --------------------------------
        taug_d = cp.empty((nc, nl, NGPTLW), dtype=cp.float32)
        fracs_d = cp.empty((nc, nl, NGPTLW), dtype=cp.float32)
        total = nc * nl
        blocks = (total * 16 + 127) // 128
        _gpu_kernel("rlw_taumol_batched")(
            (blocks, 16), (128,),
            (i32(nc), i32(nl), laytrop_d, fs_d, isv_d, wx_d,
             K["chi"], K["oneminus"], K["bandptrs"], taug_d, fracs_d))
        if _stage_probe is not None:
            _stage_probe("bands")
        # FV/IV close over the two slabs, so they are released by
        # rebinding rather than `del`: a deleted closure cell is a NameError
        # for any later FV/IV call, and the undefined-name gate cannot tell
        # that none follows (tests/test_no_undefined_names.py).
        fs_d = isv_d = None
        del wx_d

        # ---- taut --------------------------------------------------------
        taut_d = cp.empty((nc, nl, NGPTLW), dtype=cp.float32)
        total = nc * nl * NGPTLW
        _gpu_kernel("rlw_taut")(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), taug_d, taua_d, K["ngb"], taut_d))
        del taug_d, taua_d

        # ---- rtrnmc pipeline ----------------------------------------------
        secdiff = cp.empty((nc, NBNDLW), dtype=cp.float32)
        _gpu_kernel("rlw_rtrn_secdiff")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), pwvcm_d, K["a0"], K["a1"], K["a2"], secdiff))

        odcld = cp.empty((nc, nl, NGPTLW), dtype=cp.float32)
        efclfrac = cp.empty((nc, nl, NGPTLW), dtype=cp.float32)
        icldlyr = cp.zeros((nc, nl), dtype=cp.int32)
        total = nc * nl
        _gpu_kernel("rlw_rtrn_prol_gpoints")(
            (total,), (128,),
            (i32(nc), i32(nl), cldfmc_d, taucmc_d, secdiff, K["ngb"],
             odcld, efclfrac, icldlyr))
        del taucmc_d

        radld_p = scratch.take("radld_p", (nc, nl, NGPTLW), dtype=cp.float32)
        radclrd_p = scratch.take("radclrd_p", (nc, nl, NGPTLW), dtype=cp.float32)
        radlu_p = scratch.take("radlu_p", (nc, nl, NGPTLW), dtype=cp.float32)
        radclru_p = scratch.take("radclru_p", (nc, nl, NGPTLW), dtype=cp.float32)
        iclddn_p = scratch.take("iclddn_p", (nc, nl, NGPTLW), dtype=cp.uint8)
        radlu_sfc = scratch.take("radlu_sfc", (nc, NGPTLW), dtype=cp.float32)
        radclru_sfc = scratch.take("radclru_sfc", (nc, NGPTLW), dtype=cp.float32)
        total = nc * NGPTLW
        _gpu_kernel("rlw_rtrn_march_coalesced", nl)(
            ((total + 127) // 128,), (128,),
            (i32(nc), i32(nl), cldfmc_d, odcld, efclfrac, icldlyr,
             secdiff, emis_d, planklay, planklev, plankbnd,
             fracs_d, taut_d, K["tau_tbl"], K["exp_tbl"], K["tfn_tbl"],
             K["bpade"], K["ngb"],
             radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
             radlu_sfc, radclru_sfc))
        if _stage_probe is not None:
            _stage_probe("march")
        del (cldfmc_d, odcld, efclfrac, fracs_d, taut_d, planklay,
             planklev, plankbnd, secdiff, icldlyr)

        totuflux = cp.empty((nc, nl + 1), dtype=cp.float32)
        totdflux = cp.empty((nc, nl + 1), dtype=cp.float32)
        totuclfl = cp.empty((nc, nl + 1), dtype=cp.float32)
        totdclfl = cp.empty((nc, nl + 1), dtype=cp.float32)
        total = nc * (nl + 1)
        _gpu_kernel("rlw_rtrn_accum_rows")(
            ((total + 63) // 64,), (64,),
            (i32(nc), i32(nl), radld_p, radclrd_p, radlu_p, radclru_p,
             iclddn_p, radlu_sfc, radclru_sfc,
             K["ngs"], K["delwave"], WTDIFF,
             totuflux, totdflux, totuclfl, totdclfl))
        del (radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
             radlu_sfc, radclru_sfc)

        fnet = cp.empty((nc, nl + 1), dtype=cp.float32)
        fnetc = cp.empty((nc, nl + 1), dtype=cp.float32)
        htr = cp.empty((nc, nl + 1), dtype=cp.float32)
        htrc = cp.empty((nc, nl + 1), dtype=cp.float32)
        _gpu_kernel("rlw_rtrn_final")(
            ((nc + 63) // 64,), (64,),
            (i32(nc), i32(nl), plev_d, K["fluxfac"], K["heatfac"],
             totuflux, totdflux, totuclfl, totdclfl,
             fnet, fnetc, htr, htrc))

        uflx[rows] = totuflux
        dflx[rows] = totdflux
        uflxc[rows] = totuclfl
        dflxc[rows] = totdclfl
        hr[rows] = htr[:, :nl]
        hrc[rows] = htrc[:, :nl]
        # Free every per-chunk transient before the next iteration so no
        # stale slab inflates the next chunk's high-water (keeps
        # lw_batched_vram_bytes an upper bound; data movement only).
        del (totuflux, totdflux, totuclfl, totdclfl, fnet, fnetc, htr,
             htrc)

    # MPAS: module_ra_rrtmg_lw.F:2775-3036 (cldprmc stop). The first nonzero chunk word refuses with
    # the same message and type; inside a forecast it is read off the step path, at most one step
    # later (deferred_checks_v842), elsewhere at once. THE BREAKAGE THIS PREVENTS: an immediate read
    # per solver call was a full stream sync per wrapper batch (G2).
    from .deferred_checks_v842 import defer
    defer(err_flags, "rlw_cldprmc device abort, code {value} "
          "(bounds violation, mirrors the Fortran stop)", reduce="first", error=ValueError)
    return {
        "uflx": uflx, "dflx": dflx, "hr": hr,
        "uflxc": uflxc, "dflxc": dflxc, "hrc": hrc,
        "uflxcln": cp.zeros((ncol, nl + 1), dtype=cp.float32),
        "dflxcln": cp.zeros((ncol, nl + 1), dtype=cp.float32),
    }


def mcica_subcol_device(a,surface,tauc,tables,*,icld,irng,permuteseed,idcor,julday):
    """MPAS: module_ra_rrtmg_lw.F:2099-2697, general stochastic boundary.

    Packed a=(columns,12,layers) has the same layer fields as the wrapper;
    tauc=(columns,16,layers). Nonzero irng selects the native shared MT
    stream, including its g-point/column/layer draw order. KISS preserves
    its separate pressure-dependent stream for each column.
    """
    import cupy as cp
    nc,fields,nl=a.shape
    if fields!=12 or surface.shape!=(nc,6) or tauc.shape!=(nc,16,nl):
        raise ValueError('MPAS McICA boundary dimensions do not agree')
    if not 0<=icld<=5:raise ValueError('MCICA_SUBCOL: INVALID ICLD (MPAS F:2195-2196)')
    for array in (a,surface,tauc):
        if not isinstance(array,cp.ndarray) or array.dtype!=cp.float32 or not array.flags.c_contiguous:
            raise TypeError('McICA requires contiguous device float32; prevents incorrect raw indexing')
    if irng==0 and icld>0 and nl>=2 and bool(cp.any(a[:,0,0]<a[:,0,1]).item()):
        raise ValueError('MCICA_SUBCOL: KISSVEC SEED GENERATOR REQUIRES PMID FROM BOTTOM FOUR LAYERS (MPAS F:2433-2434)')
    outputs={name:cp.zeros((nc,nl,140),dtype=cp.float32)
             for name in ('cloud','liquid','ice','snow','optical_depth')}
    outputs['radii']=cp.zeros((nc,3,nl),dtype=cp.float32)
    if not nc or icld==0:return dict(outputs,irng=int(irng))
    rng=int(irng!=0)
    draws=cp.empty((140,nc,nl),dtype=cp.float32) if rng else cp.empty(0,dtype=cp.float32)
    draws2=cp.empty_like(draws)
    if rng:_gpu_kernel('mpas_mcica_mt_draws')((1,),(1,),
        (nc,nl,icld,permuteseed,draws,draws2))
    cp.cuda.get_current_stream().wait_event(tables['ready'])
    _gpu_kernel('mpas_mcica_batch')(((nc+31)//32,),(32,),
        (nc,nl,icld,idcor,julday,surface,a,rng,permuteseed,draws,draws2,tauc,tables['ngb'],
         outputs['optical_depth'],outputs['cloud'],outputs['liquid'],outputs['ice'],outputs['snow'],outputs['radii']))
    return dict(outputs,irng=rng)


MODEL_FIELDS = ('p3d','t3d','dz8w','qv3d','qc3d','qi3d','qs3d',
                'cldfra3d','re_cloud','re_ice','re_snow','pi3d')
SURFACE_FIELDS = ('tsk','emiss','xland','xice','snow','xlat')
FLUX_FIELDS = ('glw','olr','lwcf','lwupt','lwuptc','lwdnt','lwdntc',
               'lwupb','lwupbc','lwdnb','lwdnbc')


def radiation_lw_from_mpas(state):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:297-611.

    Pack MPAS arrays (levels,cells) into bottom-to-top CUDA columns.
    Field names follow the pristine driver call. No thermodynamic field
    is recomputed; the physics driver supplies its hydrated pressure,
    interface temperature, radiation condensates, and ozone climatology.
    """
    import cupy as cp
    model = cp.stack([cp.asarray(state[name], dtype=cp.float32).T
                      for name in MODEL_FIELDS], axis=1)
    interfaces = cp.stack([cp.asarray(state[name], dtype=cp.float32).T
                           for name in ('p8w','t8w')], axis=1)
    surface = cp.stack([cp.asarray(state[name], dtype=cp.float32)
                       for name in SURFACE_FIELDS], axis=1)
    return (cp.ascontiguousarray(model),cp.ascontiguousarray(interfaces),
            cp.ascontiguousarray(surface))


def vinterp_ozn_device(pmid, pin, ozone):
    """MPAS: module_ra_rrtmg_vinterp.F:21-179.

    Inputs: pmid (columns,layers), pin increasing pressure, ozone
    (columns,climatology levels). Units must agree as in MPAS.
    """
    import cupy as cp
    nc,nl=pmid.shape
    nz=pin.size
    # Linear device indexing needs binary32 contiguous buffers. These device
    # copies preserve values and avoid reading strides as contiguous memory.
    pmid=cp.ascontiguousarray(pmid,dtype=cp.float32)
    pin=cp.ascontiguousarray(pin,dtype=cp.float32)
    ozone=cp.ascontiguousarray(ozone,dtype=cp.float32)
    if pin.ndim!=1 or ozone.shape!=(nc,nz):
        raise ValueError('vinterp_ozn expects pin(levels) and ozone(columns,levels)')
    out=cp.empty_like(pmid)
    if not nc or not nl:return out
    upper=cp.zeros(nc,dtype=cp.int32)
    # MPAS: module_ra_rrtmg_vinterp.F:114-120. starts[k] is the common search
    # start of level k: min(nz-1, every column's upper) before that level. The
    # first (top) level sees the zeroed uppers; each level reduces the next's.
    starts=cp.full(nl,nz-1,dtype=cp.int32)
    starts[nl-1]=min(nz-1,0)
    kernel=_gpu_kernel('mpas_vinterp_level_chain')
    grid=((nc+127)//128,)
    for k in range(nl-1,-1,-1):
        kernel(grid,(128,),
            (np.int32(nc),np.int32(nl),np.int32(nz),np.int32(k),
             pmid,pin,ozone,starts,upper,out))
    return out


#: Layer-count histogram width: wrapper counts are nm + max(NINT(ptop/400 Pa),1), so every real
#: top pressure lands in a few bins above nm+1; anything outside falls back to a full read.
_LAYER_HISTOGRAM_BINS = 1024


def _layer_groups(counts, nm, *, need_members=False):
    """MPAS: module_ra_rrtmg_lw.F:11817-11821. Columns grouped by wrapper layer count.

    Returns (layers, device column indices, host column list or None) per group, ascending in
    layers, each group's columns in ascending column order. One small host read (the histogram).
    Per-column words do not depend on grouping order or batch width (independent KISS state per
    column, column-independent solver), so this matches the host grouping it replaces.
    """
    import cupy as cp
    nc = int(counts.size)
    if not nc:
        return []
    base = int(nm) + 1
    hist = cp.zeros(_LAYER_HISTOGRAM_BINS + 1, dtype=cp.int32)
    _gpu_kernel('mpas_layer_histogram')(((nc + 255) // 256,), (256,),
        (np.int32(nc), np.int32(base), np.int32(_LAYER_HISTOGRAM_BINS), counts, hist))
    order = cp.argsort(counts, kind='stable').astype(cp.int32)
    sizes = hist.get()  # launch metadata: group shapes
    if sizes[_LAYER_HISTOGRAM_BINS]:
        layers = counts.get().tolist()
        found = {}
        for c, nl in enumerate(layers):
            found.setdefault(nl, []).append(c)
        return [(nl, cp.asarray(members, dtype=cp.int32), members if need_members else None)
                for nl, members in sorted(found.items())]
    groups, offset = [], 0
    for bin_index in np.flatnonzero(sizes[:_LAYER_HISTOGRAM_BINS]).tolist():
        size = int(sizes[bin_index])
        indices = order[offset:offset + size]
        groups.append((base + bin_index, indices, indices.get().tolist() if need_members else None))
        offset += size
    return groups


_XORSHIFT_JUMPS = {}


def _xorshift_step_columns():
    """MPAS: module_ra_rrtmg_lw.F:2705-2737 (seed 2 of kissvec): the step's GF(2) columns."""
    columns = []
    for i in range(32):
        v = 1 << i
        v ^= (v << 13) & 0xffffffff
        v ^= v >> 17
        v ^= (v << 5) & 0xffffffff
        columns.append(v)
    return columns


def _xorshift_apply(columns, v):
    out, i = 0, 0
    while v:
        if v & 1:
            out ^= columns[i]
        v >>= 1
        i += 1
    return out


def _xorshift_compose(outer, inner):
    return [_xorshift_apply(outer, column) for column in inner]


def xorshift_jump_table(draws, gpoints):
    """Device table (gpoints, 32): the KISS xorshift advanced g*draws steps, for each g-point g.

    Built and uploaded once per device, draw count and g-point count, then reused every call."""
    import cupy as cp
    key = (int(cp.cuda.Device().id), int(draws), int(gpoints))
    table = _XORSHIFT_JUMPS.get(key)
    if table is None:
        identity = [1 << i for i in range(32)]
        power, base, n = identity, _xorshift_step_columns(), int(draws)
        while n:  # the step to the power draws, by squaring
            if n & 1:
                power = _xorshift_compose(base, power)
            base = _xorshift_compose(base, base)
            n >>= 1
        rows, current = [], identity
        for _ in range(int(gpoints)):
            rows.append(current)
            current = _xorshift_compose(power, current)
        table = cp.asarray(np.asarray(rows, dtype=np.uint32))
        _XORSHIFT_JUMPS[key] = table
    return table


def mcica_draws_per_gpoint(overlap, nl):
    """MPAS: module_ra_rrtmg_lw.F:2466-2629 / module_ra_rrtmg_sw.F:1775-1940: KISS draws per g-point."""
    overlap = int(overlap)
    if overlap == 3:
        return 1
    return 2 * int(nl) if overlap >= 4 else int(nl)


def _lw_mcica_wrapper(ng, nl, cldovrlp, idcor, julday, surface, a):
    """MPAS: module_ra_rrtmg_lw.F:12175-12185,2099-2737. Wrapper McICA (IRNG=0, seed=150).

    Every word of the four (columns, layers, 140) slabs is written when an overlap is selected;
    overlap 0 leaves them zero exactly as native's early return over cleared arrays.
    """
    import cupy as cp
    shape = (ng, nl, 140)
    if not cldovrlp:
        return tuple(cp.zeros(shape, dtype=cp.float32) for _ in range(4))
    outputs = tuple(cp.empty(shape, dtype=cp.float32) for _ in range(4))
    seeds = cp.empty((ng, 4), dtype=cp.uint32)
    weights = cp.empty((ng, nl) if cldovrlp >= 4 else (1,), dtype=cp.float32)
    _gpu_kernel('mpas_lw_mcica_seeds')(((ng + 127) // 128,), (128,),
        (np.int32(ng), np.int32(nl), np.int32(cldovrlp), np.int32(idcor), np.int32(julday),
         np.int32(150), surface, a, seeds, weights))
    draws = mcica_draws_per_gpoint(cldovrlp, nl)
    threads = ng * 140
    _gpu_kernel('mpas_lw_mcica_gpoint')(((threads + 127) // 128,), (128,),
        (np.int32(ng), np.int32(nl), np.int32(cldovrlp), np.int32(draws), a, seeds, weights,
         xorshift_jump_table(draws, 140), *outputs))
    return outputs


#: Wrapper storage beside the solver: McICA, packed inputs and gas profiles.
#: The solver's live slabs are priced separately by lw_batched_vram_bytes.
_LW_WRAPPER_COLUMN_WORDS_PER_LAYER = 5 * 140 + 12 + 2 + 16 + 2 * 12 + 8
_LW_WRAPPER_BUDGET_BYTES = 1 << 30


def _lw_wrapper_chunk(ncol, nlayers):
    """Bound the complete wrapper and solver batch, including allocator headroom.

    The former budget counted only the wrapper, then passed its entire batch
    to the solver. Its radiance and optical-depth slabs were additional live
    storage, so the advertised 1 GiB bound did not bound a radiation call.
    Column width changes storage only: each column has independent KISS state.
    The budget and the compact ceiling are divided among the same-host ranks
    on this card (woof.hex.device_share names the breakage).
    """
    import cupy as cp
    from ..device_share import ranks_on_device
    ranks = ranks_on_device(cp=cp)
    layers = max(1, int(nlayers))
    quantum = 256
    wrapper_column = 4 * _LW_WRAPPER_COLUMN_WORDS_PER_LAYER * layers
    solver = _legacy.lw_batched_vram_bytes(quantum, layers)
    per_column = wrapper_column + (solver + quantum - 1) // quantum
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    available = (int(free_bytes) + int(pool.free_bytes())) // ranks
    limit = int(pool.get_limit())
    if limit > 0:
        available = min(available, max(0, limit - int(pool.used_bytes())))
    budget = min(_LW_WRAPPER_BUDGET_BYTES // ranks, available // 8)
    width = max(quantum, budget // per_column // quantum * quantum)
    from ..device_memory_release import shared_card_memory
    ceiling = max(quantum, 1024 // ranks // quantum * quantum) if shared_card_memory() else max(1, ncol)
    return int(min(max(1, ncol), ceiling, width))


def rrtmg_lwrad_device(model, interfaces, surface, tables, *, icloud,
                       cldovrlp, idcor, julday, has_reqc=0, has_reqi=0,
                       has_reqs=0, o3input=0, pin=None, ozone=None,
                       has_qc3d=True, has_qi3d=True, has_qs3d=True,
                       has_cldfra3d=True, column_chunk=None, retain_profiles=True):
    """MPAS: module_ra_rrtmg_lw.F:11574-12305.

    Device-only MPAS wrapper, retaining variable layer counts per column.
    model=(columns,12,model layers); interfaces=(columns,2,layers+1);
    surface=(columns,6). tables is the radiation backend's initialized
    device table mapping for solve_device. Output tendencies/radii are
    column major; full profiles are a list because MPAS columns differ
    in their number of layers. retain_profiles=False omits these optional
    per-column views and returns profiles=None; fluxes, radii and tendencies
    are unchanged. Only integer layer counts cross to host
    to select launch shapes; all field processing stays on the device.
    Optional argument presence is retained separately from packed values:
    native ignores cldfra3d unless at least one condensate is present
    (MPAS: module_ra_rrtmg_lw.F:11801-11806).
    """
    import cupy as cp
    # MPAS: module_ra_rrtmg_lw.F:11810-12185. Each wrapper column has
    # independent KISS state (IRNG=0, seed=150). Bound the complete wrapper,
    # including its four 140-g-point McICA buffers, before solver chunking.
    # Defect: whole-layer-group allocation exhausted the shared card on the
    # forecasts in scratch/committed-lw-eight-suite-v10/{06,07}-job.log.
    nc,nfields,nm=model.shape
    if column_chunk is None:
        # Bounded by bytes, not a fixed 256 columns: 256 split a 216,187-cell mesh into 845 wrapper
        # batches whose host launch work made every radiation step host-bound (U-MPASPHYS-1). Words
        # are batch-invariant (independent per-column KISS state), so only memory sets the width.
        chunk = _lw_wrapper_chunk(nc, nm + 32)  # added top layers: generous upper bound
    else:
        chunk = int(column_chunk)
    if chunk < 1:
        raise ValueError('column_chunk must be >= 1; prevents empty wrapper launches')
    if nm < 1:
        raise ValueError('MPAS top extension requires at least one model layer')
    if nfields!=12 or interfaces.shape!=(nc,2,nm+1) or surface.shape!=(nc,6):
        raise ValueError('MPAS longwave boundary dimensions do not agree')
    for name,array in (('model',model),('interfaces',interfaces),('surface',surface)):
        if not isinstance(array,cp.ndarray) or array.dtype!=cp.float32 or not array.flags.c_contiguous:
            raise TypeError(f'{name} must be contiguous device float32; prevents incorrect raw indexing')
    if not 0<=cldovrlp<=5:
        raise ValueError('MCICA_SUBCOL: INVALID ICLD (MPAS F:2203-2206)')
    if o3input==2 and (pin is None or ozone is None):
        raise ValueError('MPAS climatology interpolation requires pin and ozone')
    counts=cp.empty(nc,dtype=cp.int32)
    _gpu_kernel('mpas_layer_counts')(((nc+31)//32,),(32,),
        (np.int32(nc),np.int32(nm),interfaces,counts))
    # Launch metadata, not weather fields: one read of a small layer-count
    # histogram instead of every column's count, and the column lists stay on
    # the device. THE BREAKAGE THIS PREVENTS: per-batch host index lists were
    # uploaded from pageable memory, a full stream sync per wrapper batch (G2).
    groups=_layer_groups(counts,nm,need_members=retain_profiles)
    tendency=cp.empty((nc,nm),dtype=cp.float32)
    radii=cp.empty((nc,3,nm),dtype=cp.float32)
    flux=cp.empty((nc,len(FLUX_FIELDS)),dtype=cp.float32)
    profiles=[None]*nc if retain_profiles else None
    i32=np.int32
    presence = (int(bool(has_qc3d)) | (int(bool(has_qi3d)) << 1) |
                (int(bool(has_qs3d)) << 2) | (int(bool(has_cldfra3d)) << 3))
    inflag=5 if has_reqs else 4 if has_reqi else 3 if has_reqc else 2
    iceflag=5 if has_reqs else 4 if has_reqi else 3
    dummy=cp.zeros(2,dtype=cp.float32)
    constant_gases={}
    batches=((nl,indices[first:first+chunk],
              None if members is None else members[first:first+chunk])
             for nl,indices,members in groups
             for first in range(0,int(indices.size),chunk))
    for nl,indices,members in batches:
        ng=int(indices.size)
        m=cp.ascontiguousarray(model[indices]);lev_in=cp.ascontiguousarray(interfaces[indices])
        s=cp.ascontiguousarray(surface[indices])
        oz=dummy if ozone is None else cp.ascontiguousarray(ozone[indices])
        a=cp.empty((ng,12,nl),dtype=cp.float32)
        lev=cp.empty((ng,2,nl+1),dtype=cp.float32)
        grid=((ng+31)//32,)
        _gpu_kernel('mpas_lw_prepare')(grid,(32,),
            (i32(ng),i32(nm),i32(nl),i32(icloud),i32(has_reqc),i32(has_reqi),i32(has_reqs),
             i32(o3input),i32(0 if pin is None else pin.size),i32(presence),m,lev_in,
             s,dummy if pin is None else pin,oz,a,lev))
        cloud,liquid,ice,snow=_lw_mcica_wrapper(ng,nl,cldovrlp,idcor,julday,s,a)
        # The solver runs the whole wrapper batch as one chunk (column_chunk=ng): its per-column
        # kernels (setcoef, inatm, final) otherwise launched on ~2,300 columns at a time, a few
        # threads per SM. Chunk width is workspace shape only (solve_device's contract).
        # Read-only gas profiles (the solver copies them into its own slabs): one set per shape
        # per call, not eight fills per batch.
        gases=constant_gases.get((ng,nl))
        if gases is None:
            gases=constant_gases[(ng,nl)]={value:cp.full((ng,nl),value,dtype=cp.float32)
                for value in (379.e-6,1774.e-9,319.e-9,.209488,.251e-9,.538e-9,.169e-9,.093e-9)}
        const=gases.__getitem__
        out=solve_device(ng,nl,cldovrlp,a[:,0,:],lev[:,0,:],a[:,1,:],lev[:,1,:],
            s[:,0],a[:,3,:],a[:,4,:],const(379.e-6),const(1774.e-9),
            const(319.e-9),const(.209488),const(.251e-9),const(.538e-9),
            const(.169e-9),const(.093e-9),cp.broadcast_to(s[:,1:2],(ng,16)).copy(),
            inflag,iceflag,1,cloud,cp.zeros_like(cloud),ice,liquid,snow,
            a[:,10,:],a[:,9,:],a[:,11,:],cp.zeros((ng,nl,16),dtype=cp.float32),
            tables,column_chunk=ng,mcica_layout='column')
        tendency_group=cp.empty((ng,nm),dtype=cp.float32)
        radii_group=cp.empty((ng,3,nm),dtype=cp.float32)
        flux_group=cp.empty((ng,len(FLUX_FIELDS)),dtype=cp.float32)
        _gpu_kernel('mpas_lw_outputs')(grid,(32,),
            (i32(ng),i32(nm),i32(nl),i32(icloud),m,a,out['uflx'],out['dflx'],
             out['uflxc'],out['dflxc'],out['hr'],tendency_group,radii_group,flux_group))
        tendency[indices]=tendency_group;radii[indices]=radii_group;flux[indices]=flux_group
        if profiles is not None:
            for row,c in enumerate(members):
                profiles[c]={name:array[row:row+1] for name,array in out.items()}
        # Requested profile views retain their output buffers. Otherwise
        # retire them with the inputs and McICA transients before the next
        # wrapper batch; output consumption was queued on the same stream.
        del indices,m,lev_in,s,oz,a,lev,cloud,liquid,ice,snow,out
        del tendency_group,radii_group,flux_group
    result={name:flux[:,j] for j,name in enumerate(FLUX_FIELDS)}
    result.update(rthratenlw=tendency,rre_cloud=radii[:,0],rre_ice=radii[:,1],
                  rre_snow=radii[:,2],mpas_nlay=counts-nm,profiles=profiles)
    return result


def radiation_lw_to_mpas(result):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:613-719.

    Return MPAS level-major tendencies and diagnostic radii, and surface
    fluxes under their pristine driver names. Driver registration awaits
    the integrator-owned f4 seam; this module does not change a default.
    """
    import cupy as cp
    out={name:result[name] for name in FLUX_FIELDS}
    for name in ('rthratenlw','rre_cloud','rre_ice','rre_snow'):
        out[name]=cp.ascontiguousarray(result[name].T)
    return out


def load_tables_device(path, *, decoder='rw_mpas_rrtmg_lw_tables'):
    """MPAS: mpas_atmphys_rrtmg_lwinit.F:59-1593.

    Rust decodes native Fortran records and rewrites word byte order.
    Python views its output buffer and uploads it; CUDA reduces g-points.
    The decoder is supplied by rw-mpas, including the release bridge path.
    """
    import json
    import subprocess
    import cupy as cp
    output=subprocess.run([str(decoder),str(path)],check=True,
                          stdout=subprocess.PIPE).stdout
    header,payload=output.split(b'\n',1)
    manifest=json.loads(header)
    if manifest['format']!='mpas-rrtmg-lw-raw-f32-v1':
        raise ValueError('Rust table transport version mismatch')
    raw={}
    for field in manifest['arrays']:
        view=np.ndarray((field['words'],),dtype='<f4',buffer=payload,
                        offset=field['offset'])
        raw.setdefault(field['module'],{})[field['name']]=cp.asarray(view)
    return initialize_device(raw)



def _radiation_lw_options(options, radius_flags=None):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:847-865;
    MPAS: mpas_atmphys_manager.F:747-759. Metadata only, no array maths.
    """
    from types import SimpleNamespace
    overlap = str(options['config_radt_cld_overlap']).strip()
    cldovrlp = {'none': 0, 'random': 1, 'maximum_random': 2,
                'maximum': 3, 'exponential': 4, 'exponential_random': 5}.get(overlap, 0)
    idcor = int(cldovrlp >= 4 and
                str(options['config_radt_cld_dcorrlen']).strip() == 'latitude_varying')
    if radius_flags is None:
        enabled = (bool(options['config_microp_re']) and
                   str(options['config_microp_scheme']).strip() in
                   ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6') and
                   str(options['config_radt_lw_scheme']).strip() == 'rrtmg_lw' and
                   str(options['config_radt_sw_scheme']).strip() == 'rrtmg_sw')
        radius_flags = (int(enabled),) * 3
    return SimpleNamespace(cldovrlp=cldovrlp, idcor=idcor,
            o3input=2 if options['config_o3climatology'] else 0,
            has_reqc=int(radius_flags[0]), has_reqi=int(radius_flags[1]),
            has_reqs=int(radius_flags[2]))


def load_native_lw_resources(resources, device):
    """MPAS: mpas_atmphys_rrtmg_lwinit.F:44-1593. Rust record transport.

    The f4 immutable packet authenticates both the native asset and the
    explicit decoder executable. Device initialization is the column-proven
    whole-module path; the device argument is f4's resident CUDA provider.
    """
    from .table_resources_v842 import certified_path
    asset = certified_path(resources['table'])
    decoder = certified_path(resources['decoder'])
    return load_tables_device(asset, decoder=decoder)


def bind_native_pools(provider, call):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:297-719,805-884.

    Consume the native allocator/cloudiness carrier views from f4. No p21
    arithmetic is repeated here; qvrad/qcrad/qirad/qsrad have already been
    produced by the selected native preparation calls before this slot.
    """
    aliases = {'pres_hyd': 'pres_hyd_p', 'pres2_hyd': 'pres2_hyd_p',
               't': 't_p', 't2': 't2_p', 'dz': 'dz_p', 'pi': 'pi_p',
               'xlat': 'xlat_p'}
    prepared = {name: provider.prepared(native_name)
                for name, native_name in aliases.items()}
    radiation = {name: provider.prepared(name + '_p').T
                 for name in ('qvrad', 'qcrad', 'qirad', 'qsrad')}
    return {'nCellsSolve': provider.session.layout.dimensions['nCellsSolve'],
            'prepared': prepared, 'radshared': radiation,
            'diag_physics': provider['diag_physics'],
            'sfc_input': provider['sfc_input'], 'atm_input': provider['atm_input'],
            'tend_physics': provider['tend_physics'],
            'rrtmg_lw_tables': provider.rrtmg_lw_tables,
            # The production driver consumes pool writes and discards the
            # slot result (driver_v842.py:1216-1217). Avoid per-column views.
            'retain_lw_profiles': False,
            # MPAS: mpas_atmphys_vars.F:806-807 (native PARAMETER).
            'icloud': 1,
            # MPAS: mpas_atmphys_manager.F:204-212 (current native calendar).
            'julday': int(provider.native.get_table('mpas_atmphys_manager.julday')[0][()]),
            'radius_flags': tuple(int(provider['scalar_state'][name])
                                 for name in ('has_reqc', 'has_reqi', 'has_reqs'))}


def radiation_lw_slot(call):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:297-719,805-884.

    f4 SlotCall: prepared pres_hyd,pres2_hyd,t,t2,dz,pi,xlat; p21
    radshared radiation carriers [cell,level]; native diagnostic, surface
    and tendency pools [level,cell]. Immutable initialized tables live in
    pools['rrtmg_lw_tables']; mutable numerical state stays in MPAS pools.
    Only solve cells are written, preserving resident halos.
    """
    import cupy as cp
    RADIUS_SCHEMES = ('mp_thompson', 'mp_thompson_aerosols', 'mp_wsm6')
    pools=call.pools
    nc=int(pools['nCellsSolve'])
    prep=pools['prepared'];diag=pools['diag_physics']
    rad=pools['radshared'];sfc=pools['sfc_input']
    opts=_radiation_lw_options(call.options, pools.get("radius_flags"))
    # MPAS: mpas_atmphys_driver_radiation_lw.F:374-378,480-498.
    # Native globals share kme storage, including a spare interface level;
    # the driver reads model fields only through kte.
    nl=diag['cldfrac'].shape[0]
    state={target:prep[source][:(nl+1 if target in ('p8w','t8w') else nl),:nc] for target,source in
           (('p3d','pres_hyd'),('p8w','pres2_hyd'),('t3d','t'),
            ('t8w','t2'),('dz8w','dz'),('pi3d','pi'))}
    for target,source in (('qv3d','qvrad'),('qc3d','qcrad'),
                          ('qi3d','qirad'),('qs3d','qsrad')):
        state[target]=rad[source][:nc,:nl].T
    state['cldfra3d']=diag['cldfrac'][:,:nc]
    radii_enabled=(str(call.options['config_microp_scheme']).strip() in
                   RADIUS_SCHEMES and bool(call.options['config_microp_re']))
    for target in ('re_cloud','re_ice','re_snow'):
        state[target]=(diag[target][:,:nc] if radii_enabled else
                       cp.zeros_like(state['p3d']))
    for target,source in (('tsk','skintemp'),('xland','xland'),
                          ('xice','xice'),('snow','snow')):
        state[target]=sfc[source][:nc]
    state['emiss']=diag['sfc_emiss'][:nc]
    state['xlat']=prep['xlat'][:nc]
    pin=pools['atm_input']['pin'] if opts.o3input==2 else None
    ozone=(cp.ascontiguousarray(diag['o3clim'][:,:nc].T)
           if opts.o3input==2 else None)
    if opts.o3input==2:
        # This diagnostic uses model pressures, separately from the extended
        # solver columns, exactly as radiation_lw_from_MPAS does.
        pressure=cp.ascontiguousarray(state['p3d'].T)
        pressure_mb=cp.empty_like(pressure)
        _gpu_kernel('mpas_pressure_mb')(((pressure.size+127)//128,),(128,),
            (np.int32(pressure.size),pressure,pressure_mb))
        diag['o3vmr'][:,:nc]=vinterp_ozn_device(pressure_mb,pin,ozone).T
    packed=radiation_lw_from_mpas(state)
    result=rrtmg_lwrad_device(*packed,pools['rrtmg_lw_tables'],
        icloud=int(pools['icloud']),julday=int(pools['julday']),
        cldovrlp=opts.cldovrlp,idcor=opts.idcor,has_reqc=opts.has_reqc,
        has_reqi=opts.has_reqi,has_reqs=opts.has_reqs,o3input=opts.o3input,
        pin=pin,ozone=ozone,retain_profiles=pools.get('retain_lw_profiles', True))
    out=radiation_lw_to_mpas(result)
    for name in FLUX_FIELDS:
        diag['olrtoa' if name=='olr' else name][:nc]=out[name]
    pools['tend_physics']['rthratenlw'][:,:nc]=out['rthratenlw']
    if str(call.options['config_microp_scheme']).strip() in RADIUS_SCHEMES:
        for name in ('rre_cloud','rre_ice','rre_snow'):
            diag[name][:,:nc]=out[name] if radii_enabled else np.float32(0)
    return result


def initialize_radiation_lw(call):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:776-797.

    f4 initialization hook for the selected longwave backend. The framework
    supplies the table asset path and the installed Rust decoder path.
    Constants can be rebuilt on restart; held fluxes and tendencies remain
    in their native restart pools.
    """
    pools=call.pools
    tables=load_tables_device(pools['rrtmg_lw_table_path'],
                             decoder=pools['rrtmg_lw_table_decoder'])
    pools['rrtmg_lw_tables']=tables
    return tables


def register_hex_cuda(registry):
    """MPAS: mpas_atmphys_driver_radiation_lw.F:841-883; f4 slot hook."""
    registry.register('radiation_lw','rrtmg_lw','hex-cuda',radiation_lw_slot)

# MPAS: module_ra_rrtmg_lw.F:8331-9982 (cmbgb1-cmbgb16).
_REDUCTIONS = ((1, 'kg01/ka', 'kao', 65, False, True), (1, 'kg01/kb', 'kbo', 235, False, True), (1, 'kg01/selfref', 'selfrefo', 10, False, True), (1, 'kg01/forref', 'forrefo', 4, False, True), (1, 'kg01/ka_mn2', 'kao_mn2', 19, False, True), (1, 'kg01/kb_mn2', 'kbo_mn2', 19, False, True), (1, 'kg01/fracrefa', 'fracrefao', 1, False, False), (1, 'kg01/fracrefb', 'fracrefbo', 1, False, False), (2, 'kg02/ka', 'kao', 65, False, True), (2, 'kg02/kb', 'kbo', 235, False, True), (2, 'kg02/selfref', 'selfrefo', 10, False, True), (2, 'kg02/forref', 'forrefo', 4, False, True), (2, 'kg02/fracrefa', 'fracrefao', 1, False, False), (2, 'kg02/fracrefb', 'fracrefbo', 1, False, False), (3, 'kg03/ka', 'kao', 585, False, True), (3, 'kg03/kb', 'kbo', 1175, False, True), (3, 'kg03/ka_mn2o', 'kao_mn2o', 171, False, True), (3, 'kg03/kb_mn2o', 'kbo_mn2o', 95, False, True), (3, 'kg03/selfref', 'selfrefo', 10, False, True), (3, 'kg03/forref', 'forrefo', 4, False, True), (3, 'kg03/fracrefa', 'fracrefao', 9, True, False), (3, 'kg03/fracrefb', 'fracrefbo', 5, True, False), (4, 'kg04/ka', 'kao', 585, False, True), (4, 'kg04/kb', 'kbo', 1175, False, True), (4, 'kg04/selfref', 'selfrefo', 10, False, True), (4, 'kg04/forref', 'forrefo', 4, False, True), (4, 'kg04/fracrefa', 'fracrefao', 9, True, False), (4, 'kg04/fracrefb', 'fracrefbo', 5, True, False), (5, 'kg05/ka', 'kao', 585, False, True), (5, 'kg05/kb', 'kbo', 1175, False, True), (5, 'kg05/ka_mo3', 'kao_mo3', 171, False, True), (5, 'kg05/selfref', 'selfrefo', 10, False, True), (5, 'kg05/forref', 'forrefo', 4, False, True), (5, 'kg05/fracrefa', 'fracrefao', 9, True, False), (5, 'kg05/fracrefb', 'fracrefbo', 5, True, False), (5, 'kg05/ccl4', 'ccl4o', 1, False, True), (6, 'kg06/ka', 'kao', 65, False, True), (6, 'kg06/ka_mco2', 'kao_mco2', 19, False, True), (6, 'kg06/selfref', 'selfrefo', 10, False, True), (6, 'kg06/forref', 'forrefo', 4, False, True), (6, 'kg06/fracrefa', 'fracrefao', 1, False, False), (6, 'kg06/cfc11adj', 'cfc11adjo', 1, False, True), (6, 'kg06/cfc12', 'cfc12o', 1, False, True), (7, 'kg07/ka', 'kao', 585, False, True), (7, 'kg07/kb', 'kbo', 235, False, True), (7, 'kg07/ka_mco2', 'kao_mco2', 171, False, True), (7, 'kg07/kb_mco2', 'kbo_mco2', 19, False, True), (7, 'kg07/selfref', 'selfrefo', 10, False, True), (7, 'kg07/forref', 'forrefo', 4, False, True), (7, 'kg07/fracrefa', 'fracrefao', 9, True, False), (7, 'kg07/fracrefb', 'fracrefbo', 1, False, False), (8, 'kg08/ka', 'kao', 65, False, True), (8, 'kg08/kb', 'kbo', 235, False, True), (8, 'kg08/selfref', 'selfrefo', 10, False, True), (8, 'kg08/forref', 'forrefo', 4, False, True), (8, 'kg08/ka_mco2', 'kao_mco2', 19, False, True), (8, 'kg08/kb_mco2', 'kbo_mco2', 19, False, True), (8, 'kg08/ka_mo3', 'kao_mo3', 19, False, True), (8, 'kg08/ka_mn2o', 'kao_mn2o', 19, False, True), (8, 'kg08/kb_mn2o', 'kbo_mn2o', 19, False, True), (8, 'kg08/fracrefa', 'fracrefao', 1, False, False), (8, 'kg08/fracrefb', 'fracrefbo', 1, False, False), (8, 'kg08/cfc12', 'cfc12o', 1, False, True), (8, 'kg08/cfc22adj', 'cfc22adjo', 1, False, True), (9, 'kg09/ka', 'kao', 585, False, True), (9, 'kg09/kb', 'kbo', 235, False, True), (9, 'kg09/ka_mn2o', 'kao_mn2o', 171, False, True), (9, 'kg09/kb_mn2o', 'kbo_mn2o', 19, False, True), (9, 'kg09/selfref', 'selfrefo', 10, False, True), (9, 'kg09/forref', 'forrefo', 4, False, True), (9, 'kg09/fracrefa', 'fracrefao', 9, True, False), (9, 'kg09/fracrefb', 'fracrefbo', 1, False, False), (10, 'kg10/ka', 'kao', 65, False, True), (10, 'kg10/kb', 'kbo', 235, False, True), (10, 'kg10/selfref', 'selfrefo', 10, False, True), (10, 'kg10/forref', 'forrefo', 4, False, True), (10, 'kg10/fracrefa', 'fracrefao', 1, False, False), (10, 'kg10/fracrefb', 'fracrefbo', 1, False, False), (11, 'kg11/ka', 'kao', 65, False, True), (11, 'kg11/kb', 'kbo', 235, False, True), (11, 'kg11/ka_mo2', 'kao_mo2', 19, False, True), (11, 'kg11/kb_mo2', 'kbo_mo2', 19, False, True), (11, 'kg11/selfref', 'selfrefo', 10, False, True), (11, 'kg11/forref', 'forrefo', 4, False, True), (11, 'kg11/fracrefa', 'fracrefao', 1, False, False), (11, 'kg11/fracrefb', 'fracrefbo', 1, False, False), (12, 'kg12/ka', 'kao', 585, False, True), (12, 'kg12/selfref', 'selfrefo', 10, False, True), (12, 'kg12/forref', 'forrefo', 4, False, True), (12, 'kg12/fracrefa', 'fracrefao', 9, True, False), (13, 'kg13/ka', 'kao', 585, False, True), (13, 'kg13/ka_mco2', 'kao_mco2', 171, False, True), (13, 'kg13/ka_mco', 'kao_mco', 171, False, True), (13, 'kg13/kb_mo3', 'kbo_mo3', 19, False, True), (13, 'kg13/selfref', 'selfrefo', 10, False, True), (13, 'kg13/forref', 'forrefo', 4, False, True), (13, 'kg13/fracrefb', 'fracrefbo', 1, False, False), (13, 'kg13/fracrefa', 'fracrefao', 9, True, False), (14, 'kg14/ka', 'kao', 65, False, True), (14, 'kg14/kb', 'kbo', 235, False, True), (14, 'kg14/selfref', 'selfrefo', 10, False, True), (14, 'kg14/forref', 'forrefo', 4, False, True), (14, 'kg14/fracrefa', 'fracrefao', 1, False, False), (14, 'kg14/fracrefb', 'fracrefbo', 1, False, False), (15, 'kg15/ka', 'kao', 585, False, True), (15, 'kg15/ka_mn2', 'kao_mn2', 171, False, True), (15, 'kg15/selfref', 'selfrefo', 10, False, True), (15, 'kg15/forref', 'forrefo', 4, False, True), (15, 'kg15/fracrefa', 'fracrefao', 9, True, False), (16, 'kg16/ka', 'kao', 585, False, True), (16, 'kg16/kb', 'kbo', 235, False, True), (16, 'kg16/selfref', 'selfrefo', 10, False, True), (16, 'kg16/forref', 'forrefo', 4, False, True), (16, 'kg16/fracrefb', 'fracrefbo', 1, False, False), (16, 'kg16/fracrefa', 'fracrefao', 9, True, False))
_STATIC_ARRAYS = (('chi', 413, False), ('totplnk', 2896, False), ('totplk16', 181, False), ('preflog', 59, False), ('tref', 59, False), ('absice1', 10, False), ('absice2', 688, False), ('absice3', 736, False), ('absice0', 2, False), ('absliq1', 928, False), ('ngb', 140, True), ('ngs', 16, True), ('delwave', 16, False), ('ngc', 16, True), ('ngn', 140, True), ('ngm', 256, True), ('wt', 16, False), ('a0', 16, False), ('a1', 16, False), ('a2', 16, False))
_STATIC_SCALARS = {'oneminus': 0.9999989867210388, 'bpade': 3.5971224308013916, 'absliq0': 0.09036140143871307, 'grav': 9.806599617004395, 'avogad': 6.02214208470173e+23, 'fluxfac': 62831.85546875, 'heatfac': 8.434944152832031}
_INIT_SOURCE = r"""
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_3E8E5604_DEFINED
#define HEX_DIVISOR_3E8E5604_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_3e8e5604 = 0.278f;
#else
static const float hex_divisor_3e8e5604 = 0.278f;
#endif
#endif

// MPAS: module_ra_rrtmg_lw.F:3935-4801,8137-8328,9985-10623.
// Generated immutable DATA arrays; initialization replay checks every word.
extern "C" __device__ const float mpas_static_chi[413] = {
0x1.335d240000000p-6f,0x1.743e960000000p-12f,0x1.0328880000000p-25f,0x1.5798ee0000000p-22f,0x1.421f600000000p-23f,0x1.c8571c0000000p-20f,0x1.ac08320000000p-3f,0x1.9085f40000000p-7f,
0x1.743e960000000p-12f,0x1.2a49160000000p-25f,0x1.5798ee0000000p-22f,0x1.3338120000000p-23f,0x1.c8571c0000000p-20f,0x1.ac08320000000p-3f,0x1.8210e60000000p-8f,0x1.743e960000000p-12f,
0x1.6cdfea0000000p-25f,0x1.5798ee0000000p-22f,0x1.215a1a0000000p-23f,0x1.c8503e0000000p-20f,0x1.ac08320000000p-3f,0x1.6abde40000000p-9f,0x1.743e960000000p-12f,0x1.c532440000000p-25f,
0x1.5798ee0000000p-22f,0x1.187b9c0000000p-23f,0x1.c5c3680000000p-20f,0x1.ac08320000000p-3f,0x1.70b49e0000000p-10f,0x1.743e960000000p-12f,0x1.1f85b40000000p-24f,0x1.5798ee0000000p-22f,
0x1.12ba420000000p-23f,0x1.bf823e0000000p-20f,0x1.ac08320000000p-3f,0x1.8e4d360000000p-11f,0x1.743e960000000p-12f,0x1.7638720000000p-24f,0x1.5738ba0000000p-22f,0x1.02839a0000000p-23f,
0x1.b6eb380000000p-20f,0x1.ac08320000000p-3f,0x1.97a4f80000000p-12f,0x1.743e960000000p-12f,0x1.e93d5e0000000p-24f,0x1.5292800000000p-22f,0x1.cfc5440000000p-24f,0x1.b0209e0000000p-20f,
0x1.ac08320000000p-3f,0x1.5ae9300000000p-13f,0x1.743e960000000p-12f,0x1.50a9380000000p-23f,0x1.463c280000000p-22f,0x1.94bfc60000000p-24f,0x1.a27daa0000000p-20f,0x1.ac08320000000p-3f,
0x1.37f8ea0000000p-15f,0x1.743e960000000p-12f,0x1.d3e4cc0000000p-23f,0x1.3bea940000000p-22f,0x1.5565fa0000000p-24f,0x1.95dfda0000000p-20f,0x1.ac08320000000p-3f,0x1.f5bd4a0000000p-18f,
0x1.743e960000000p-12f,0x1.5c5aa40000000p-22f,0x1.31f6760000000p-22f,0x1.07a69e0000000p-24f,0x1.8bb3620000000p-20f,0x1.ac08320000000p-3f,0x1.211e500000000p-18f,0x1.743e960000000p-12f,
0x1.f44c9c0000000p-22f,0x1.291d780000000p-22f,0x1.8762180000000p-25f,0x1.8224f80000000p-20f,0x1.ac08320000000p-3f,0x1.bf33360000000p-19f,0x1.743e960000000p-12f,0x1.30f9920000000p-21f,
0x1.1c3aee0000000p-22f,0x1.1f8c4e0000000p-25f,0x1.77dd020000000p-20f,0x1.ac08320000000p-3f,0x1.ae052a0000000p-19f,0x1.743e960000000p-12f,0x1.75b3220000000p-21f,0x1.04c21a0000000p-22f,
0x1.af86220000000p-26f,0x1.6c58f20000000p-20f,0x1.ac08320000000p-3f,0x1.a8620e0000000p-19f,0x1.743e960000000p-12f,0x1.2c459c0000000p-20f,0x1.c201540000000p-23f,0x1.40c3ee0000000p-26f,
0x1.6074ac0000000p-20f,0x1.ac08320000000p-3f,0x1.b4879c0000000p-19f,0x1.743e960000000p-12f,0x1.d8edfa0000000p-20f,0x1.7142840000000p-23f,0x1.ed07240000000p-27f,0x1.4fddd00000000p-20f,
0x1.ac08320000000p-3f,0x1.cb5fa40000000p-19f,0x1.743e960000000p-12f,0x1.384fac0000000p-19f,0x1.2741ee0000000p-23f,0x1.ad87f60000000p-27f,0x1.3935e20000000p-20f,0x1.ac08320000000p-3f,
0x1.e70c9e0000000p-19f,0x1.743e960000000p-12f,0x1.8cf9cc0000000p-19f,0x1.e6b4a80000000p-24f,0x1.aa4d200000000p-27f,0x1.1b47aa0000000p-20f,0x1.ac08320000000p-3f,0x1.06b7c00000000p-18f,
0x1.743e960000000p-12f,0x1.eb24980000000p-19f,0x1.aefffe0000000p-24f,0x1.cee4f80000000p-27f,0x1.f4cc6e0000000p-21f,0x1.ac08320000000p-3f,0x1.166bd80000000p-18f,0x1.743e960000000p-12f,
0x1.345d800000000p-18f,0x1.880c860000000p-24f,0x1.fff5ca0000000p-27f,0x1.b407cc0000000p-21f,0x1.ac08320000000p-3f,0x1.211c980000000p-18f,0x1.743e960000000p-12f,0x1.64f2020000000p-18f,
0x1.6f0f900000000p-24f,0x1.143c300000000p-26f,0x1.93efb20000000p-21f,0x1.ac08320000000p-3f,0x1.2a18fa0000000p-18f,0x1.743e960000000p-12f,0x1.9016f20000000p-18f,0x1.59280e0000000p-24f,
0x1.26bce20000000p-26f,0x1.7bb7d20000000p-21f,0x1.ac08320000000p-3f,0x1.3336020000000p-18f,0x1.743e960000000p-12f,0x1.b4f7480000000p-18f,0x1.3b215e0000000p-24f,0x1.3fd2080000000p-26f,
0x1.6636040000000p-21f,0x1.ac08320000000p-3f,0x1.3bfeda0000000p-18f,0x1.743e960000000p-12f,0x1.da06020000000p-18f,0x1.1b5c480000000p-24f,0x1.596b200000000p-26f,0x1.50cb940000000p-21f,
0x1.ac08320000000p-3f,0x1.41bd720000000p-18f,0x1.743e960000000p-12f,0x1.0217340000000p-17f,0x1.e15f100000000p-25f,0x1.74034c0000000p-26f,0x1.3aa0ba0000000p-21f,0x1.ac08320000000p-3f,
0x1.46ccd00000000p-18f,0x1.743e960000000p-12f,0x1.15151a0000000p-17f,0x1.9480020000000p-25f,0x1.8cc9480000000p-26f,0x1.2609f00000000p-21f,0x1.ac08320000000p-3f,0x1.4a940a0000000p-18f,
0x1.743e960000000p-12f,0x1.2432de0000000p-17f,0x1.57665a0000000p-25f,0x1.a228740000000p-26f,0x1.1461920000000p-21f,0x1.ac08320000000p-3f,0x1.4d52b20000000p-18f,0x1.743e960000000p-12f,
0x1.285e980000000p-17f,0x1.1b49960000000p-25f,0x1.b88b1c0000000p-26f,0x1.02cdd20000000p-21f,0x1.ac08320000000p-3f,0x1.4f4bc80000000p-18f,0x1.743e960000000p-12f,0x1.246c6c0000000p-17f,
0x1.bfc6b00000000p-26f,0x1.cfaf460000000p-26f,0x1.e292600000000p-22f,0x1.ac08320000000p-3f,0x1.5314ba0000000p-18f,0x1.743e960000000p-12f,0x1.0f99820000000p-17f,0x1.69e9400000000p-26f,
0x1.e8ded00000000p-26f,0x1.bfc5960000000p-22f,0x1.ac08320000000p-3f,0x1.580a520000000p-18f,0x1.743e960000000p-12f,0x1.ebf4780000000p-18f,0x1.1d0c640000000p-26f,0x1.0071a40000000p-25f,
0x1.9cf60c0000000p-22f,0x1.ac08320000000p-3f,0x1.6057780000000p-18f,0x1.743e960000000p-12f,0x1.a776b20000000p-18f,0x1.bf42be0000000p-27f,0x1.09cc600000000p-25f,0x1.79f5060000000p-22f,
0x1.ac08320000000p-3f,0x1.6782460000000p-18f,0x1.743e960000000p-12f,0x1.682fcc0000000p-18f,0x1.5ab0940000000p-27f,0x1.1313520000000p-25f,0x1.56cd860000000p-22f,0x1.ac08320000000p-3f,
0x1.6dcdc00000000p-18f,0x1.743e960000000p-12f,0x1.2cd7a20000000p-18f,0x1.05fd540000000p-27f,0x1.1c55de0000000p-25f,0x1.3395860000000p-22f,0x1.ac08320000000p-3f,0x1.6ff53a0000000p-18f,
0x1.743e960000000p-12f,0x1.01a33c0000000p-18f,0x1.a4480c0000000p-28f,0x1.264a8a0000000p-25f,0x1.11c85e0000000p-22f,0x1.ac08320000000p-3f,0x1.7119480000000p-18f,0x1.743e960000000p-12f,
0x1.b898b60000000p-19f,0x1.40ba400000000p-28f,0x1.2fbd760000000p-25f,0x1.e2a0200000000p-23f,0x1.ac08320000000p-3f,0x1.7119480000000p-18f,0x1.743e960000000p-12f,0x1.7af6b80000000p-19f,
0x1.c3954a0000000p-29f,0x1.38a5da0000000p-25f,0x1.a4a62c0000000p-23f,0x1.ac08320000000p-3f,0x1.6dfc220000000p-18f,0x1.743e960000000p-12f,0x1.4e485e0000000p-19f,0x1.877b2a0000000p-29f,
0x1.407d8e0000000p-25f,0x1.88e1d40000000p-23f,0x1.ac08320000000p-3f,0x1.6ab40a0000000p-18f,0x1.743e960000000p-12f,0x1.2283a80000000p-19f,0x1.525fec0000000p-29f,0x1.4848120000000p-25f,
0x1.6efbc60000000p-23f,0x1.ac08320000000p-3f,0x1.676bf20000000p-18f,0x1.743e960000000p-12f,0x1.ed84c20000000p-20f,0x1.1d44ae0000000p-29f,0x1.5010640000000p-25f,0x1.5510380000000p-23f,
0x1.ac08320000000p-3f,0x1.608b020000000p-18f,0x1.743e960000000p-12f,0x1.be16060000000p-20f,0x1.fded0e0000000p-30f,0x1.62ca420000000p-25f,0x1.4da2100000000p-23f,0x1.ac08320000000p-3f,
0x1.592e600000000p-18f,0x1.743e960000000p-12f,0x1.940c8e0000000p-20f,0x1.c76af00000000p-30f,0x1.76f7c40000000p-25f,0x1.48a6a20000000p-23f,0x1.ac08320000000p-3f,0x1.51d3760000000p-18f,
0x1.743e960000000p-12f,0x1.69fc360000000p-20f,0x1.90e1c60000000p-30f,0x1.8b27780000000p-25f,0x1.43ab320000000p-23f,0x1.ac08320000000p-3f,0x1.47435a0000000p-18f,0x1.743e960000000p-12f,
0x1.415ef60000000p-20f,0x1.6ab38e0000000p-30f,0x1.affaae0000000p-25f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.3b454e0000000p-18f,0x1.743e960000000p-12f,0x1.195fc20000000p-20f,
0x1.4bd5260000000p-30f,0x1.dc470c0000000p-25f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.2f458c0000000p-18f,0x1.743e960000000p-12f,0x1.e2c95a0000000p-21f,0x1.2cefb40000000p-30f,
0x1.0449b40000000p-24f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.22fbea0000000p-18f,0x1.743e960000000p-12f,0x1.99dd060000000p-21f,0x1.1245ee0000000p-30f,0x1.281eee0000000p-24f,
0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.161cd00000000p-18f,0x1.743e960000000p-12f,0x1.5f02f60000000p-21f,0x1.fff7320000000p-31f,0x1.6748580000000p-24f,0x1.421f600000000p-23f,
0x1.ac08320000000p-3f,0x1.093c000000000p-18f,0x1.743e960000000p-12f,0x1.2428e80000000p-21f,0x1.db62860000000p-31f,0x1.a670aa0000000p-24f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,
0x1.f8b9cc0000000p-19f,0x1.743e960000000p-12f,0x1.d29af20000000p-22f,0x1.b6cdda0000000p-31f,0x1.e596c80000000p-24f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.df73dc0000000p-19f,
0x1.743e960000000p-12f,0x1.8711480000000p-22f,0x1.9d13560000000p-31f,0x1.3d18f20000000p-23f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.c664e40000000p-19f,0x1.743e960000000p-12f,
0x1.4ef16a0000000p-22f,0x1.8858200000000p-31f,0x1.9b12420000000p-23f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.ad55ee0000000p-19f,0x1.743e960000000p-12f,0x1.16d18c0000000p-22f,
0x1.739cea0000000p-31f,0x1.f90b920000000p-23f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.9446f80000000p-19f,0x1.743e960000000p-12f,0x1.bd635c0000000p-23f,0x1.5ee1b40000000p-31f,
0x1.2b82720000000p-22f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.7b537e0000000p-19f,0x1.73f0be0000000p-12f,0x1.9b28400000000p-23f,0x1.4f3f200000000p-31f,0x1.77d2040000000p-22f,
0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.626dc20000000p-19f,0x1.737aa20000000p-12f,0x1.9fd6b80000000p-23f,0x1.42282c0000000p-31f,0x1.d2beae0000000p-22f,0x1.421f600000000p-23f,
0x1.ac08320000000p-3f,0x1.4988080000000p-19f,0x1.7307340000000p-12f,0x1.a4852e0000000p-23f,0x1.350fce0000000p-31f,0x1.16d70c0000000p-21f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,
0x1.309edc0000000p-19f,0x1.7291180000000p-12f,0x1.a933a60000000p-23f,0x1.27f8d80000000p-31f,0x1.444ec00000000p-21f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.22fbea0000000p-18f,
0x1.743e960000000p-12f,0x1.99dd060000000p-21f,0x1.1245ee0000000p-30f,0x1.281eee0000000p-24f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f,0x1.161cd00000000p-18f,0x1.743e960000000p-12f,
0x1.5f02f60000000p-21f,0x1.fff7320000000p-31f,0x1.6748580000000p-24f,0x1.421f600000000p-23f,0x1.ac08320000000p-3f
};
extern "C" __device__ const float mpas_static_totplnk[2896] = {
0x1.8cd4000000000p-20f,0x1.92d0720000000p-20f,0x1.98d3c40000000p-20f,0x1.9eddf40000000p-20f,0x1.a4ef020000000p-20f,0x1.ab06f20000000p-20f,0x1.b125c00000000p-20f,0x1.b74b6c0000000p-20f,
0x1.bd77f80000000p-20f,0x1.c3ab640000000p-20f,0x1.c9ec8e0000000p-20f,0x1.d02db80000000p-20f,0x1.d675c20000000p-20f,0x1.dccb880000000p-20f,0x1.e321500000000p-20f,0x1.e97df80000000p-20f,
0x1.efe17e0000000p-20f,0x1.f64be20000000p-20f,0x1.fcc4060000000p-20f,0x1.019e140000000p-19f,0x1.04da260000000p-19f,0x1.081d180000000p-19f,0x1.0b63780000000p-19f,0x1.0ea9da0000000p-19f,
0x1.11f71a0000000p-19f,0x1.1544580000000p-19f,0x1.1895080000000p-19f,0x1.1be9280000000p-19f,0x1.1f40b60000000p-19f,0x1.229bb40000000p-19f,0x1.25f6b20000000p-19f,0x1.2958900000000p-19f,
0x1.2cba6e0000000p-19f,0x1.301fbc0000000p-19f,0x1.3385080000000p-19f,0x1.36f1340000000p-19f,0x1.3a5d620000000p-19f,0x1.3dccfe0000000p-19f,0x1.413c9a0000000p-19f,0x1.44b3140000000p-19f,
0x1.4829900000000p-19f,0x1.4ba37a0000000p-19f,0x1.4f1d660000000p-19f,0x1.529e300000000p-19f,0x1.561efa0000000p-19f,0x1.599fc40000000p-19f,0x1.5d276c0000000p-19f,0x1.60af160000000p-19f,
0x1.6436be0000000p-19f,0x1.67c5480000000p-19f,0x1.6b53d00000000p-19f,0x1.6ee2580000000p-19f,0x1.7277c00000000p-19f,0x1.760d280000000p-19f,0x1.79a2900000000p-19f,0x1.7d3b660000000p-19f,
0x1.80d7ae0000000p-19f,0x1.8477640000000p-19f,0x1.88171a0000000p-19f,0x1.8bb6d20000000p-19f,0x1.8f59f80000000p-19f,0x1.93008c0000000p-19f,0x1.96a7220000000p-19f,0x1.9a51280000000p-19f,
0x1.9dfe9c0000000p-19f,0x1.a1ac120000000p-19f,0x1.a559860000000p-19f,0x1.a90dda0000000p-19f,0x1.acbec00000000p-19f,0x1.b073140000000p-19f,0x1.b42ad80000000p-19f,0x1.b7e29c0000000p-19f,
0x1.bb9dce0000000p-19f,0x1.bf59020000000p-19f,0x1.c317a60000000p-19f,0x1.c6d6480000000p-19f,0x1.ca985a0000000p-19f,0x1.ce5a6e0000000p-19f,0x1.d21ff00000000p-19f,0x1.d5e5720000000p-19f,
0x1.d9ae640000000p-19f,0x1.dd77560000000p-19f,0x1.e143b60000000p-19f,0x1.e510180000000p-19f,0x1.e8dfea0000000p-19f,0x1.ecafba0000000p-19f,0x1.f07f8c0000000p-19f,0x1.f452cc0000000p-19f,
0x1.f8297c0000000p-19f,0x1.fbfcbc0000000p-19f,0x1.ffd6dc0000000p-19f,0x1.01d6c60000000p-18f,0x1.03c3d60000000p-18f,0x1.05b29e0000000p-18f,0x1.07a1660000000p-18f,0x1.09902e0000000p-18f,
0x1.0b80ae0000000p-18f,0x1.0d712c0000000p-18f,0x1.0f63640000000p-18f,0x1.11559c0000000p-18f,0x1.1347d20000000p-18f,0x1.153bc20000000p-18f,0x1.172fb20000000p-18f,0x1.1923a00000000p-18f,
0x1.1b19480000000p-18f,0x1.1d0eee0000000p-18f,0x1.1f064c0000000p-18f,0x1.20fdac0000000p-18f,0x1.22f50a0000000p-18f,0x1.24ee220000000p-18f,0x1.26e7380000000p-18f,0x1.28e04e0000000p-18f,
0x1.2adb1c0000000p-18f,0x1.2cd5ea0000000p-18f,0x1.2ed2720000000p-18f,0x1.30cd400000000p-18f,0x1.32c9c60000000p-18f,0x1.34c8040000000p-18f,0x1.36c6420000000p-18f,0x1.38c4800000000p-18f,
0x1.3ac2be0000000p-18f,0x1.3cc2b40000000p-18f,0x1.3ec2aa0000000p-18f,0x1.40c29e0000000p-18f,0x1.42c44c0000000p-18f,0x1.44c5fa0000000p-18f,0x1.46c9600000000p-18f,0x1.48cb0e0000000p-18f,
0x1.4ace720000000p-18f,0x1.4cd1d80000000p-18f,0x1.4ed6f60000000p-18f,0x1.50dc120000000p-18f,0x1.52e1300000000p-18f,0x1.54e64c0000000p-18f,0x1.56ed220000000p-18f,0x1.58f3f60000000p-18f,
0x1.5afacc0000000p-18f,0x1.5d03580000000p-18f,0x1.5f0a2e0000000p-18f,0x1.6112ba0000000p-18f,0x1.631cfe0000000p-18f,0x1.65258c0000000p-18f,0x1.672fd00000000p-18f,0x1.693a140000000p-18f,
0x1.6b46120000000p-18f,0x1.6d50560000000p-18f,0x1.6f5c520000000p-18f,0x1.716a060000000p-18f,0x1.7376040000000p-18f,0x1.7583b80000000p-18f,0x1.77916c0000000p-18f,0x1.799f200000000p-18f,
0x1.7bacd40000000p-18f,0x1.7dbc400000000p-18f,0x1.7fcbac0000000p-18f,0x1.81db180000000p-18f,0x1.83ea840000000p-18f,0x1.85fba80000000p-18f,0x1.880ccc0000000p-18f,0x1.8a1df00000000p-18f,
0x1.8c2f140000000p-18f,0x1.8e41f00000000p-18f,0x1.9053120000000p-18f,0x1.9265ee0000000p-18f,0x1.947a820000000p-18f,0x1.968d5e0000000p-18f,0x1.98a1f20000000p-18f,0x1.9ab6840000000p-18f,
0x1.9ccb180000000p-18f,0x1.9edfac0000000p-18f,0x1.a0f43e0000000p-18f,0x1.a30a8a0000000p-18f,0x1.a520d60000000p-18f,0x1.a737200000000p-18f,0x1.a94f240000000p-18f,0x1.ab656e0000000p-18f,
0x1.ad7bba0000000p-18f,0x1.af93be0000000p-18f,0x1.b1abc00000000p-18f,0x1.b3c57c0000000p-18f,0x1.b5dd7e0000000p-18f,0x1.0ff3b40000000p-19f,0x1.1698820000000p-19f,0x1.1d4e7e0000000p-19f,
0x1.24191a0000000p-19f,0x1.2afbc00000000p-19f,0x1.31ef960000000p-19f,0x1.38fb7a0000000p-19f,0x1.40188a0000000p-19f,0x1.474da80000000p-19f,0x1.4e93f40000000p-19f,0x1.55f24e0000000p-19f,
0x1.5d61d60000000p-19f,0x1.64e5fa0000000p-19f,0x1.6c7ebe0000000p-19f,0x1.7428ae0000000p-19f,0x1.7beaac0000000p-19f,0x1.83bdda0000000p-19f,0x1.8ba5a40000000p-19f,0x1.93a20a0000000p-19f,
0x1.9bafa00000000p-19f,0x1.a3d1d40000000p-19f,0x1.ac05340000000p-19f,0x1.b44d340000000p-19f,0x1.bca6600000000p-19f,0x1.c5142c0000000p-19f,0x1.cd96940000000p-19f,0x1.d626ba0000000p-19f,
0x1.deceee0000000p-19f,0x1.e784e00000000p-19f,0x1.f04f700000000p-19f,0x1.f92b2e0000000p-19f,0x1.010dc60000000p-18f,0x1.058cd20000000p-18f,0x1.0a162e0000000p-18f,0x1.0ea8220000000p-18f,
0x1.1342ac0000000p-18f,0x1.17e5cc0000000p-18f,0x1.1c933c0000000p-18f,0x1.21478c0000000p-18f,0x1.2604720000000p-18f,0x1.2acba60000000p-18f,0x1.2f99ba0000000p-18f,0x1.3470660000000p-18f,
0x1.394fa80000000p-18f,0x1.3e37800000000p-18f,0x1.43263a0000000p-18f,0x1.481f400000000p-18f,0x1.4d1f280000000p-18f,0x1.5227a40000000p-18f,0x1.5737020000000p-18f,0x1.5c4ef60000000p-18f,
0x1.616f800000000p-18f,0x1.6696ea0000000p-18f,0x1.6bc6ea0000000p-18f,0x1.70ff820000000p-18f,0x1.763efa0000000p-18f,0x1.7b85500000000p-18f,0x1.80d2860000000p-18f,0x1.862a0a0000000p-18f,
0x1.8b86b60000000p-18f,0x1.90ebfa0000000p-18f,0x1.96581c0000000p-18f,0x1.9bcb1e0000000p-18f,0x1.a146b60000000p-18f,0x1.a6c92e0000000p-18f,0x1.ac52840000000p-18f,0x1.b1e2ba0000000p-18f,
0x1.b779d00000000p-18f,0x1.bd17c40000000p-18f,0x1.c2be500000000p-18f,0x1.c86a020000000p-18f,0x1.ce1c940000000p-18f,0x1.d3d7be0000000p-18f,0x1.d9980e0000000p-18f,0x1.df5f3e0000000p-18f,
0x1.e52d4c0000000p-18f,0x1.eb023c0000000p-18f,0x1.f0de080000000p-18f,0x1.f6befe0000000p-18f,0x1.fca88a0000000p-18f,0x1.014b9e0000000p-17f,0x1.04458c0000000p-17f,0x1.0743c40000000p-17f,
0x1.0a44920000000p-17f,0x1.0d47f20000000p-17f,0x1.104f9e0000000p-17f,0x1.1359dc0000000p-17f,0x1.1666b00000000p-17f,0x1.1976f20000000p-17f,0x1.1c89ca0000000p-17f,0x1.1fa0100000000p-17f,
0x1.22b9c60000000p-17f,0x1.25d5320000000p-17f,0x1.28f4100000000p-17f,0x1.2c165c0000000p-17f,0x1.2f3b3e0000000p-17f,0x1.3262b20000000p-17f,0x1.358d960000000p-17f,0x1.38ba300000000p-17f,
0x1.3bea3c0000000p-17f,0x1.3f1db60000000p-17f,0x1.4252ea0000000p-17f,0x1.458b8c0000000p-17f,0x1.48c5e60000000p-17f,0x1.4c03b00000000p-17f,0x1.4f440c0000000p-17f,0x1.5287da0000000p-17f,
0x1.55c9120000000p-17f,0x1.5912e40000000p-17f,0x1.5c5cb40000000p-17f,0x1.5faf1a0000000p-17f,0x1.6301820000000p-17f,0x1.6653ea0000000p-17f,0x1.69a6500000000p-17f,0x1.6cf8b80000000p-17f,
0x1.7053b60000000p-17f,0x1.73aeb40000000p-17f,0x1.77124a0000000p-17f,0x1.7a75e00000000p-17f,0x1.7dd9740000000p-17f,0x1.813d0a0000000p-17f,0x1.84a9360000000p-17f,0x1.880ccc0000000p-17f,
0x1.8b78f80000000p-17f,0x1.8eedbc0000000p-17f,0x1.92627e0000000p-17f,0x1.95ceac0000000p-17f,0x1.994c060000000p-17f,0x1.9cc0ca0000000p-17f,0x1.a03e240000000p-17f,0x1.a3bb7e0000000p-17f,
0x1.a738d80000000p-17f,0x1.aab6320000000p-17f,0x1.ae3c240000000p-17f,0x1.b1c2160000000p-17f,0x1.b548080000000p-17f,0x1.b8d6900000000p-17f,0x1.bc65180000000p-17f,0x1.bfeb0a0000000p-17f,
0x1.c382280000000p-17f,0x1.c710b20000000p-17f,0x1.caa7d00000000p-17f,0x1.ce3ef00000000p-17f,0x1.d1d6100000000p-17f,0x1.d56d300000000p-17f,0x1.d90ce60000000p-17f,0x1.dcac9c0000000p-17f,
0x1.e04c520000000p-17f,0x1.e3ec0a0000000p-17f,0x1.e794560000000p-17f,0x1.eb340e0000000p-17f,0x1.eedc5a0000000p-17f,0x1.f284a80000000p-17f,0x1.f6358e0000000p-17f,0x1.f9ddda0000000p-17f,
0x1.fd8ec00000000p-17f,0x1.009fd20000000p-16f,0x1.027c900000000p-16f,0x1.0455020000000p-16f,0x1.0631c00000000p-16f,0x1.080e7e0000000p-16f,0x1.09eb3c0000000p-16f,0x1.0bc7fa0000000p-16f,
0x1.0da4b60000000p-16f,0x1.0f85c00000000p-16f,0x1.1166ca0000000p-16f,0x1.1347d20000000p-16f,0x1.152d280000000p-16f,0x1.170e300000000p-16f,0x1.18f3860000000p-16f,0x1.1ad8da0000000p-16f,
0x1.1cbe300000000p-16f,0x1.1ea3840000000p-16f,0x1.2088d80000000p-16f,0x1.22727a0000000p-16f,0x1.245c1a0000000p-16f,0x1.2645ba0000000p-16f,0x1.282f5a0000000p-16f,0x1.2a1d460000000p-16f,
0x1.2c06e60000000p-16f,0x1.2df4d20000000p-16f,0x1.69e8fa0000000p-20f,0x1.757c280000000p-20f,0x1.8149100000000p-20f,0x1.8d4fb20000000p-20f,0x1.99900e0000000p-20f,0x1.a60a220000000p-20f,
0x1.b2bf500000000p-20f,0x1.bfae380000000p-20f,0x1.ccd83a0000000p-20f,0x1.da3bf40000000p-20f,0x1.e7db760000000p-20f,0x1.f5b5640000000p-20f,0x1.01e5340000000p-19f,0x1.090d440000000p-19f,
0x1.1053380000000p-19f,0x1.17b6620000000p-19f,0x1.1f37700000000p-19f,0x1.26d6620000000p-19f,0x1.2e92e00000000p-19f,0x1.366cec0000000p-19f,0x1.3e64dc0000000p-19f,0x1.467ab00000000p-19f,
0x1.4eae120000000p-19f,0x1.56ff580000000p-19f,0x1.5f6e820000000p-19f,0x1.67fb3a0000000p-19f,0x1.70a5d40000000p-19f,0x1.796dfe0000000p-19f,0x1.82540a0000000p-19f,0x1.8b57fc0000000p-19f,
0x1.9479220000000p-19f,0x1.9db82c0000000p-19f,0x1.a7151c0000000p-19f,0x1.b08f400000000p-19f,0x1.ba26f20000000p-19f,0x1.c3dc2e0000000p-19f,0x1.cdaf500000000p-19f,0x1.d79f500000000p-19f,
0x1.e1acda0000000p-19f,0x1.ebd79a0000000p-19f,0x1.f61fe80000000p-19f,0x1.0042880000000p-18f,0x1.0583e40000000p-18f,0x1.0ad3800000000p-18f,0x1.1031b80000000p-18f,0x1.159e8a0000000p-18f,
0x1.1b19a00000000p-18f,0x1.20a34e0000000p-18f,0x1.263b140000000p-18f,0x1.2be1720000000p-18f,0x1.3195e80000000p-18f,0x1.3758cc0000000p-18f,0x1.3d29c80000000p-18f,0x1.4308d80000000p-18f,
0x1.48f62c0000000p-18f,0x1.4ef1960000000p-18f,0x1.54faea0000000p-18f,0x1.5b122a0000000p-18f,0x1.6137aa0000000p-18f,0x1.676abe0000000p-18f,0x1.6dabbc0000000p-18f,0x1.73faa40000000p-18f,
0x1.7a574a0000000p-18f,0x1.80c1840000000p-18f,0x1.87397c0000000p-18f,0x1.8dbf060000000p-18f,0x1.9452220000000p-18f,0x1.9af2a60000000p-18f,0x1.a1a0bc0000000p-18f,0x1.a85c380000000p-18f,
0x1.af24ee0000000p-18f,0x1.b5fade0000000p-18f,0x1.bcde620000000p-18f,0x1.c3cef40000000p-18f,0x1.cacc940000000p-18f,0x1.d1d7180000000p-18f,0x1.d8eed60000000p-18f,0x1.e013a20000000p-18f,
0x1.e745240000000p-18f,0x1.ee838a0000000p-18f,0x1.f5ced00000000p-18f,0x1.fd26ce0000000p-18f,0x1.0245c20000000p-17f,0x1.05fe4c0000000p-17f,0x1.09bd460000000p-17f,0x1.0d82700000000p-17f,
0x1.114de00000000p-17f,0x1.151f800000000p-17f,0x1.18f7380000000p-17f,0x1.1cd51e0000000p-17f,0x1.20b9360000000p-17f,0x1.24a33a0000000p-17f,0x1.28936c0000000p-17f,0x1.2c89a40000000p-17f,
0x1.3085c80000000p-17f,0x1.3488040000000p-17f,0x1.38901a0000000p-17f,0x1.3c9e1c0000000p-17f,0x1.40b1de0000000p-17f,0x1.44cba60000000p-17f,0x1.48eb2e0000000p-17f,0x1.4d10760000000p-17f,
0x1.513b6c0000000p-17f,0x1.556c4e0000000p-17f,0x1.59a3320000000p-17f,0x1.5ddf400000000p-17f,0x1.6221500000000p-17f,0x1.6668880000000p-17f,0x1.6ab5c20000000p-17f,0x1.6f08240000000p-17f,
0x1.73608a0000000p-17f,0x1.77be160000000p-17f,0x1.7c21a60000000p-17f,0x1.8089820000000p-17f,0x1.84f83e0000000p-17f,0x1.896b440000000p-17f,0x1.8de44e0000000p-17f,0x1.92627e0000000p-17f,
0x1.96e5d80000000p-17f,0x1.9b6e580000000p-17f,0x1.9ffc000000000p-17f,0x1.a48faa0000000p-17f,0x1.a927a00000000p-17f,0x1.adc4be0000000p-17f,0x1.b267de0000000p-17f,0x1.b70f4c0000000p-17f,
0x1.bbbbe00000000p-17f,0x1.c06d9a0000000p-17f,0x1.c5247e0000000p-17f,0x1.c9e0880000000p-17f,0x1.cea1ba0000000p-17f,0x1.d367360000000p-17f,0x1.d831da0000000p-17f,0x1.dd01a60000000p-17f,
0x1.e1d5be0000000p-17f,0x1.e6afd80000000p-17f,0x1.eb8d640000000p-17f,0x1.f070160000000p-17f,0x1.f557ee0000000p-17f,0x1.fa44120000000p-17f,0x1.ff35600000000p-17f,0x1.02157c0000000p-16f,
0x1.0492da0000000p-16f,0x1.0712600000000p-16f,0x1.09940c0000000p-16f,0x1.0c184a0000000p-16f,0x1.0e9eae0000000p-16f,0x1.1127380000000p-16f,0x1.13b1e80000000p-16f,0x1.163f2c0000000p-16f,
0x1.18ce960000000p-16f,0x1.1b5fb80000000p-16f,0x1.1df3da0000000p-16f,0x1.2089b40000000p-16f,0x1.2321b60000000p-16f,0x1.25bbdc0000000p-16f,0x1.2858280000000p-16f,0x1.2af7080000000p-16f,
0x1.2d979e0000000p-16f,0x1.303a5c0000000p-16f,0x1.32df400000000p-16f,0x1.3586480000000p-16f,0x1.382f080000000p-16f,0x1.3ada5c0000000p-16f,0x1.3d87680000000p-16f,0x1.40369a0000000p-16f,
0x1.42e7f20000000p-16f,0x1.459b020000000p-16f,0x1.4850380000000p-16f,0x1.4b07920000000p-16f,0x1.4dc0a40000000p-16f,0x1.507bde0000000p-16f,0x1.53393c0000000p-16f,0x1.55f8520000000p-16f,
0x1.58b98e0000000p-16f,0x1.5b7c800000000p-16f,0x1.5e412c0000000p-16f,0x1.6107fe0000000p-16f,0x1.63d0f40000000p-16f,0x1.669ba40000000p-16f,0x1.69679c0000000p-16f,0x1.de193c0000000p-21f,
0x1.f0301a0000000p-21f,0x1.015d480000000p-20f,0x1.0adcda0000000p-20f,0x1.14972c0000000p-20f,0x1.1e8caa0000000p-20f,0x1.28bdf20000000p-20f,0x1.332bb20000000p-20f,0x1.3dd5ec0000000p-20f,
0x1.48bd4e0000000p-20f,0x1.53e33a0000000p-20f,0x1.5f464e0000000p-20f,0x1.6ae7ec0000000p-20f,0x1.76c9720000000p-20f,0x1.82e9820000000p-20f,0x1.8f497a0000000p-20f,0x1.9be95a0000000p-20f,
0x1.a8c9d20000000p-20f,0x1.b5eb940000000p-20f,0x1.c34dee0000000p-20f,0x1.d0f2420000000p-20f,0x1.ded7de0000000p-20f,0x1.ed00220000000p-20f,0x1.fb6b0e0000000p-20f,0x1.050bfa0000000p-19f,
0x1.0c84180000000p-19f,0x1.141e3a0000000p-19f,0x1.1bd9b20000000p-19f,0x1.23b6d40000000p-19f,0x1.2bb6540000000p-19f,0x1.33d7d60000000p-19f,0x1.3c1bb60000000p-19f,0x1.4481f20000000p-19f,
0x1.4d0a320000000p-19f,0x1.55b5260000000p-19f,0x1.5e82cc0000000p-19f,0x1.6772d00000000p-19f,0x1.7085e00000000p-19f,0x1.79bba40000000p-19f,0x1.83141c0000000p-19f,0x1.8c8fa00000000p-19f,
0x1.962e300000000p-19f,0x1.9fef740000000p-19f,0x1.a9d41c0000000p-19f,0x1.b3dc280000000p-19f,0x1.be06e80000000p-19f,0x1.c8550c0000000p-19f,0x1.d2c63c0000000p-19f,0x1.dd5ad00000000p-19f,
0x1.e812700000000p-19f,0x1.f2edcc0000000p-19f,0x1.fdec340000000p-19f,0x1.0487000000000p-18f,0x1.0a296c0000000p-18f,0x1.0fddb60000000p-18f,0x1.15a3860000000p-18f,0x1.1b7b080000000p-18f,
0x1.21643c0000000p-18f,0x1.275ef60000000p-18f,0x1.2d6b8e0000000p-18f,0x1.3389800000000p-18f,0x1.39b9500000000p-18f,0x1.3ffaa60000000p-18f,0x1.464d820000000p-18f,0x1.4cb1e40000000p-18f,
0x1.5328240000000p-18f,0x1.59afbe0000000p-18f,0x1.60490a0000000p-18f,0x1.66f3840000000p-18f,0x1.6dafdc0000000p-18f,0x1.747d620000000p-18f,0x1.7b5c6e0000000p-18f,0x1.824cd40000000p-18f,
0x1.894e940000000p-18f,0x1.9061da0000000p-18f,0x1.97867a0000000p-18f,0x1.9ebc1c0000000p-18f,0x1.a603440000000p-18f,0x1.ad5bc60000000p-18f,0x1.b4c51e0000000p-18f,0x1.bc3fd00000000p-18f,
0x1.c3cbdc0000000p-18f,0x1.cb68be0000000p-18f,0x1.d316a40000000p-18f,0x1.dad5b60000000p-18f,0x1.e2a59e0000000p-18f,0x1.ea868a0000000p-18f,0x1.f278760000000p-18f,0x1.fa7b0c0000000p-18f,
0x1.01473c0000000p-17f,0x1.0559740000000p-17f,0x1.0973d40000000p-17f,0x1.0d96a00000000p-17f,0x1.11c1c20000000p-17f,0x1.15f5360000000p-17f,0x1.1a30d60000000p-17f,0x1.1e74ca0000000p-17f,
0x1.22c1120000000p-17f,0x1.27156e0000000p-17f,0x1.2b71f40000000p-17f,0x1.2fd6b80000000p-17f,0x1.3443780000000p-17f,0x1.38b8620000000p-17f,0x1.3d35760000000p-17f,0x1.41ba860000000p-17f,
0x1.4647940000000p-17f,0x1.4adcb40000000p-17f,0x1.4f79900000000p-17f,0x1.541f040000000p-17f,0x1.58cb980000000p-17f,0x1.5d7fe60000000p-17f,0x1.623ccc0000000p-17f,0x1.67016e0000000p-17f,
0x1.6bcdca0000000p-17f,0x1.70a1e20000000p-17f,0x1.757db40000000p-17f,0x1.7a61420000000p-17f,0x1.7f4c8a0000000p-17f,0x1.843f8e0000000p-17f,0x1.8939720000000p-17f,0x1.8e3bec0000000p-17f,
0x1.9345460000000p-17f,0x1.9857360000000p-17f,0x1.9d70060000000p-17f,0x1.a28fb40000000p-17f,0x1.a7b71e0000000p-17f,0x1.ace6440000000p-17f,0x1.b21d240000000p-17f,0x1.b75bbe0000000p-17f,
0x1.bca05e0000000p-17f,0x1.c1ecb80000000p-17f,0x1.c740cc0000000p-17f,0x1.cc9c9c0000000p-17f,0x1.d1fe700000000p-17f,0x1.d767fe0000000p-17f,0x1.dcd9480000000p-17f,0x1.e251700000000p-17f,
0x1.e7d0780000000p-17f,0x1.ed56600000000p-17f,0x1.f2e3260000000p-17f,0x1.f877a80000000p-17f,0x1.fe13080000000p-17f,0x1.01da360000000p-16f,0x1.04aec60000000p-16f,0x1.0786c40000000p-16f,
0x1.0a62a20000000p-16f,0x1.0d41120000000p-16f,0x1.1023600000000p-16f,0x1.13091e0000000p-16f,0x1.15f24c0000000p-16f,0x1.18de7a0000000p-16f,0x1.1bce860000000p-16f,0x1.1ec1940000000p-16f,
0x1.21b7a40000000p-16f,0x1.24b1920000000p-16f,0x1.27ae800000000p-16f,0x1.2aaee00000000p-16f,0x1.2db2400000000p-16f,0x1.30b9100000000p-16f,0x1.33c2e20000000p-16f,0x1.36d0220000000p-16f,
0x1.39e0d40000000p-16f,0x1.3cf4180000000p-16f,0x1.400acc0000000p-16f,0x1.4324f00000000p-16f,0x1.4642140000000p-16f,0x1.4961ce0000000p-16f,0x1.4c85d20000000p-16f,0x1.4fabfc0000000p-16f,
0x1.52d5960000000p-16f,0x1.5602320000000p-16f,0x1.5931d00000000p-16f,0x1.5c646e0000000p-16f,0x1.5f9a100000000p-16f,0x1.62d2b20000000p-16f,0x1.660e560000000p-16f,0x1.694cfa0000000p-16f,
0x1.6c8f100000000p-16f,0x1.6fd34a0000000p-16f,0x1.731af60000000p-16f,0x1.7665a20000000p-16f,0x1.3223d00000000p-21f,0x1.3f530a0000000p-21f,0x1.4ce7d60000000p-21f,0x1.5ae36e0000000p-21f,
0x1.69479e0000000p-21f,0x1.7815a20000000p-21f,0x1.874efe0000000p-21f,0x1.96f5560000000p-21f,0x1.a709e80000000p-21f,0x1.b78df20000000p-21f,0x1.c883180000000p-21f,0x1.d9ea960000000p-21f,
0x1.ebc5f20000000p-21f,0x1.fe16200000000p-21f,0x1.086e740000000p-20f,0x1.120dc60000000p-20f,0x1.1be9280000000p-20f,0x1.2602620000000p-20f,0x1.3058c40000000p-20f,0x1.3aedb00000000p-20f,
0x1.45c1260000000p-20f,0x1.50d4840000000p-20f,0x1.5c27ca0000000p-20f,0x1.67bba80000000p-20f,0x1.7390d00000000p-20f,0x1.7fa7f00000000p-20f,0x1.8c01080000000p-20f,0x1.989d7a0000000p-20f,
0x1.a57c920000000p-20f,0x1.b29fb40000000p-20f,0x1.c0078e0000000p-20f,0x1.cdb3700000000p-20f,0x1.dba56a0000000p-20f,0x1.e9dccc0000000p-20f,0x1.f85a460000000p-20f,0x1.038f440000000p-19f,
0x1.0b14ca0000000p-19f,0x1.12be0a0000000p-19f,0x1.1a8b600000000p-19f,0x1.227cc80000000p-19f,0x1.2a929c0000000p-19f,0x1.32ccde0000000p-19f,0x1.3b2be20000000p-19f,0x1.43b0020000000p-19f,
0x1.4c58e60000000p-19f,0x1.5526e60000000p-19f,0x1.5e1a580000000p-19f,0x1.6733980000000p-19f,0x1.70724a0000000p-19f,0x1.79d6ca0000000p-19f,0x1.83616c0000000p-19f,0x1.8d12320000000p-19f,
0x1.96e91c0000000p-19f,0x1.a0e6800000000p-19f,0x1.ab0a620000000p-19f,0x1.b555160000000p-19f,0x1.bfc6460000000p-19f,0x1.ca5e4a0000000p-19f,0x1.d51d780000000p-19f,0x1.e003d40000000p-19f,
0x1.eb11020000000p-19f,0x1.f6455c0000000p-19f,0x1.00d09c0000000p-18f,0x1.06924c0000000p-18f,0x1.0c67be0000000p-18f,0x1.1250f20000000p-18f,0x1.184de80000000p-18f,0x1.1e5ecc0000000p-18f,
0x1.24839e0000000p-18f,0x1.2abc300000000p-18f,0x1.3108dc0000000p-18f,0x1.37694c0000000p-18f,0x1.3dde000000000p-18f,0x1.4466740000000p-18f,0x1.4b03040000000p-18f,0x1.51b3ae0000000p-18f,
0x1.5878700000000p-18f,0x1.5f514e0000000p-18f,0x1.663e180000000p-18f,0x1.6d3f280000000p-18f,0x1.7454520000000p-18f,0x1.7b7d6a0000000p-18f,0x1.82bac60000000p-18f,0x1.8a0c3e0000000p-18f,
0x1.9171ce0000000p-18f,0x1.98eba40000000p-18f,0x1.a079940000000p-18f,0x1.a81b720000000p-18f,0x1.afd1960000000p-18f,0x1.b79bd40000000p-18f,0x1.bf7a2a0000000p-18f,0x1.c76cc60000000p-18f,
0x1.cf73520000000p-18f,0x1.d78dca0000000p-18f,0x1.dfbc880000000p-18f,0x1.e7ff5e0000000p-18f,0x1.f056240000000p-18f,0x1.f8c1020000000p-18f,0x1.009ffe0000000p-17f,0x1.04e95c0000000p-17f,
0x1.093cc60000000p-17f,0x1.0d9a260000000p-17f,0x1.12017e0000000p-17f,0x1.1672cc0000000p-17f,0x1.1aee280000000p-17f,0x1.1f734e0000000p-17f,0x1.24026c0000000p-17f,0x1.289b540000000p-17f,
0x1.2d3e4a0000000p-17f,0x1.31eaf20000000p-17f,0x1.36a1940000000p-17f,0x1.3b62160000000p-17f,0x1.402c4c0000000p-17f,0x1.4500640000000p-17f,0x1.49de5c0000000p-17f,0x1.4ec61e0000000p-17f,
0x1.53b7140000000p-17f,0x1.58b2ae0000000p-17f,0x1.5db7bc0000000p-17f,0x1.62c63c0000000p-17f,0x1.67de300000000p-17f,0x1.6d00740000000p-17f,0x1.722c280000000p-17f,0x1.7761500000000p-17f,
0x1.7c9fec0000000p-17f,0x1.81e7fa0000000p-17f,0x1.873a580000000p-17f,0x1.8c954c0000000p-17f,0x1.91fa8e0000000p-17f,0x1.9769440000000p-17f,0x1.9ce16e0000000p-17f,0x1.a2622e0000000p-17f,
0x1.a7ed3c0000000p-17f,0x1.ad81be0000000p-17f,0x1.b31ed60000000p-17f,0x1.b8c63e0000000p-17f,0x1.be763c0000000p-17f,0x1.c42fae0000000p-17f,0x1.c9f2920000000p-17f,0x1.cfbee80000000p-17f,
0x1.d593d60000000p-17f,0x1.db72380000000p-17f,0x1.e15a0c0000000p-17f,0x1.e74b540000000p-17f,0x1.ed45320000000p-17f,0x1.f347a60000000p-17f,0x1.f9546a0000000p-17f,0x1.ff69c60000000p-17f,
0x1.02c3dc0000000p-16f,0x1.05d78e0000000p-16f,0x1.08ef8c0000000p-16f,0x1.0c0bd60000000p-16f,0x1.0f2cd80000000p-16f,0x1.1252940000000p-16f,0x1.157c2e0000000p-16f,0x1.18aa820000000p-16f,
0x1.1bdd8e0000000p-16f,0x1.1f147a0000000p-16f,0x1.22501e0000000p-16f,0x1.258f9e0000000p-16f,0x1.28d3da0000000p-16f,0x1.2c1c600000000p-16f,0x1.2f69320000000p-16f,0x1.32ba500000000p-16f,
0x1.360fb80000000p-16f,0x1.39696c0000000p-16f,0x1.3cc6fe0000000p-16f,0x1.40294a0000000p-16f,0x1.438f740000000p-16f,0x1.46fa560000000p-16f,0x1.4a69160000000p-16f,0x1.4ddc220000000p-16f,
0x1.51530a0000000p-16f,0x1.54ce400000000p-16f,0x1.584dc00000000p-16f,0x1.5bd18c0000000p-16f,0x1.5f59340000000p-16f,0x1.62e4bc0000000p-16f,0x1.6674fc0000000p-16f,0x1.6a08ac0000000p-16f,
0x1.6da0a60000000p-16f,0x1.25a92e0000000p-22f,0x1.349a1c0000000p-22f,0x1.441b580000000p-22f,0x1.5430320000000p-22f,0x1.64dc380000000p-22f,0x1.76222e0000000p-22f,0x1.8805ec0000000p-22f,
0x1.9a8a780000000p-22f,0x1.adb31e0000000p-22f,0x1.c1832a0000000p-22f,0x1.d5fe2e0000000p-22f,0x1.eb27340000000p-22f,0x1.0080c20000000p-21f,0x1.0bc85a0000000p-21f,0x1.176c080000000p-21f,
0x1.236d2a0000000p-21f,0x1.2fcdd20000000p-21f,0x1.3c8f3a0000000p-21f,0x1.49b3520000000p-21f,0x1.573b780000000p-21f,0x1.6529760000000p-21f,0x1.737ef20000000p-21f,0x1.823d4c0000000p-21f,
0x1.91666e0000000p-21f,0x1.a0fb9a0000000p-21f,0x1.b0feb80000000p-21f,0x1.c171280000000p-21f,0x1.d254700000000p-21f,0x1.e3aa340000000p-21f,0x1.f573f80000000p-21f,0x1.03d9b00000000p-20f,
0x1.0d34b40000000p-20f,0x1.16cc0c0000000p-20f,0x1.20a08e0000000p-20f,0x1.2ab2ea0000000p-20f,0x1.3503cc0000000p-20f,0x1.3f93ea0000000p-20f,0x1.4a633e0000000p-20f,0x1.5573dc0000000p-20f,
0x1.60c5c40000000p-20f,0x1.6c59a20000000p-20f,0x1.78302a0000000p-20f,0x1.844aba0000000p-20f,0x1.90a8a20000000p-20f,0x1.9d4bf20000000p-20f,0x1.aa33fa0000000p-20f,0x1.b762ca0000000p-20f,
0x1.c4d7b20000000p-20f,0x1.d294c20000000p-20f,0x1.e0994a0000000p-20f,0x1.eee6aa0000000p-20f,0x1.fd7ce20000000p-20f,0x1.062ea80000000p-19f,0x1.0dc3fc0000000p-19f,0x1.157ec20000000p-19f,
0x1.1d5f560000000p-19f,0x1.25660c0000000p-19f,0x1.2d933e0000000p-19f,0x1.35e6ea0000000p-19f,0x1.3e616c0000000p-19f,0x1.4703180000000p-19f,0x1.4fcc4a0000000p-19f,0x1.58bd540000000p-19f,
0x1.61d5e40000000p-19f,0x1.6b16fe0000000p-19f,0x1.7480a40000000p-19f,0x1.7e127e0000000p-19f,0x1.87cd920000000p-19f,0x1.91b1e20000000p-19f,0x1.9bbf6e0000000p-19f,0x1.a5f68c0000000p-19f,
0x1.b057960000000p-19f,0x1.bae28c0000000p-19f,0x1.c597c40000000p-19f,0x1.d077400000000p-19f,0x1.db81ae0000000p-19f,0x1.e6b6b80000000p-19f,0x1.f216b60000000p-19f,0x1.fda2540000000p-19f,
0x1.04ac740000000p-18f,0x1.0a9d900000000p-18f,0x1.10a4a80000000p-18f,0x1.16c1be0000000p-18f,0x1.1cf4a60000000p-18f,0x1.233de20000000p-18f,0x1.299d1c0000000p-18f,0x1.3012ac0000000p-18f,
0x1.369ebe0000000p-18f,0x1.3d41240000000p-18f,0x1.43fa0e0000000p-18f,0x1.4ac9780000000p-18f,0x1.51af8e0000000p-18f,0x1.58ac520000000p-18f,0x1.5fbfc40000000p-18f,0x1.66ea100000000p-18f,
0x1.6e2b360000000p-18f,0x1.7583600000000p-18f,0x1.7cf28e0000000p-18f,0x1.8478980000000p-18f,0x1.8c15d20000000p-18f,0x1.93ca3e0000000p-18f,0x1.9b95b00000000p-18f,0x1.a3787e0000000p-18f,
0x1.ab72aa0000000p-18f,0x1.b383da0000000p-18f,0x1.bbac940000000p-18f,0x1.c3ecac0000000p-18f,0x1.cc44220000000p-18f,0x1.d4b3200000000p-18f,0x1.dd397c0000000p-18f,0x1.e5d78e0000000p-18f,
0x1.ee8cfc0000000p-18f,0x1.f75a200000000p-18f,0x1.001f660000000p-17f,0x1.049d820000000p-17f,0x1.0927780000000p-17f,0x1.0dbd5e0000000p-17f,0x1.125f0a0000000p-17f,0x1.170ca60000000p-17f,
0x1.1bc5f00000000p-17f,0x1.208b400000000p-17f,0x1.255c400000000p-17f,0x1.2a39460000000p-17f,0x1.2f223e0000000p-17f,0x1.3416fa0000000p-17f,0x1.3917a60000000p-17f,0x1.3e24440000000p-17f,
0x1.433cd00000000p-17f,0x1.48613a0000000p-17f,0x1.4d917c0000000p-17f,0x1.52cd6e0000000p-17f,0x1.58157c0000000p-17f,0x1.5d69900000000p-17f,0x1.62c9ac0000000p-17f,0x1.6835ce0000000p-17f,
0x1.6dad1c0000000p-17f,0x1.73314a0000000p-17f,0x1.78c0a60000000p-17f,0x1.7e5c060000000p-17f,0x1.84036e0000000p-17f,0x1.89b6000000000p-17f,0x1.8f75740000000p-17f,0x1.9540140000000p-17f,
0x1.9b16ba0000000p-17f,0x1.a0f9660000000p-17f,0x1.a6e81a0000000p-17f,0x1.ace1f80000000p-17f,0x1.b2e7dc0000000p-17f,0x1.b8f9c80000000p-17f,0x1.bf17ba0000000p-17f,0x1.c540d60000000p-17f,
0x1.cb75fa0000000p-17f,0x1.d1b7240000000p-17f,0x1.d804540000000p-17f,0x1.de5cb00000000p-17f,0x1.e4c1120000000p-17f,0x1.eb309e0000000p-17f,0x1.f1ac320000000p-17f,0x1.f833cc0000000p-17f,
0x1.fec6900000000p-17f,0x1.02b2ae0000000p-16f,0x1.0607a80000000p-16f,0x1.0962a60000000p-16f,0x1.0cc3a80000000p-16f,0x1.102a400000000p-16f,0x1.13966c0000000p-16f,0x1.17089c0000000p-16f,
0x1.1a80600000000p-16f,0x1.1dfe280000000p-16f,0x1.2181860000000p-16f,0x1.250a7a0000000p-16f,0x1.2899700000000p-16f,0x1.2c2dfc0000000p-16f,0x1.2fc88c0000000p-16f,0x1.3368b00000000p-16f,
0x1.370e6a0000000p-16f,0x1.3ab9b80000000p-16f,0x1.3e6a9e0000000p-16f,0x1.4221860000000p-16f,0x1.45de020000000p-16f,0x1.499fa60000000p-16f,0x1.0d2f580000000p-23f,0x1.1d0bd60000000p-23f,
0x1.2da19a0000000p-23f,0x1.3ef5940000000p-23f,0x1.510e600000000p-23f,0x1.63f0ee0000000p-23f,0x1.77a34a0000000p-23f,0x1.8c2b800000000p-23f,0x1.a1902c0000000p-23f,0x1.b7d5b00000000p-23f,
0x1.cf03c00000000p-23f,0x1.e71fdc0000000p-23f,0x1.00184e0000000p-22f,0x1.0d1dc00000000p-22f,0x1.1aa3d60000000p-22f,0x1.28ad0a0000000p-22f,0x1.373d360000000p-22f,0x1.4657a40000000p-22f,
0x1.55fed00000000p-22f,0x1.6636920000000p-22f,0x1.7701f00000000p-22f,0x1.8864340000000p-22f,0x1.9a60f60000000p-22f,0x1.acfaf20000000p-22f,0x1.c035bc0000000p-22f,0x1.d414a00000000p-22f,
0x1.e89b320000000p-22f,0x1.fdcbe80000000p-22f,0x1.09d5960000000p-21f,0x1.151dde0000000p-21f,0x1.20c0b80000000p-21f,0x1.2cbf600000000p-21f,0x1.391be80000000p-21f,0x1.45d7b00000000p-21f,
0x1.52f4c60000000p-21f,0x1.6074660000000p-21f,0x1.6e58a20000000p-21f,0x1.7ca2d80000000p-21f,0x1.8b54d20000000p-21f,0x1.9a70380000000p-21f,0x1.a9f6d00000000p-21f,0x1.b9ea200000000p-21f,
0x1.ca4bcc0000000p-21f,0x1.db1d7c0000000p-21f,0x1.ec611c0000000p-21f,0x1.fe17ea0000000p-21f,0x1.0821d60000000p-20f,0x1.1173280000000p-20f,0x1.1b00320000000p-20f,0x1.24ca660000000p-20f,
0x1.2ed2720000000p-20f,0x1.3919060000000p-20f,0x1.439ed40000000p-20f,0x1.4e653a0000000p-20f,0x1.596c3a0000000p-20f,0x1.64b4800000000p-20f,0x1.703f700000000p-20f,0x1.7c0db80000000p-20f,
0x1.88200a0000000p-20f,0x1.9477120000000p-20f,0x1.a114340000000p-20f,0x1.adf6bc0000000p-20f,0x1.bb20bc0000000p-20f,0x1.c892e60000000p-20f,0x1.d64c860000000p-20f,0x1.e4505e0000000p-20f,
0x1.f29dbe0000000p-20f,0x1.009b020000000p-19f,0x1.080c9a0000000p-19f,0x1.0fa4540000000p-19f,0x1.1762e40000000p-19f,0x1.1f47960000000p-19f,0x1.2753740000000p-19f,0x1.2f872c0000000p-19f,
0x1.37e26a0000000p-19f,0x1.4065820000000p-19f,0x1.4911260000000p-19f,0x1.51e5ac0000000p-19f,0x1.5ae3160000000p-19f,0x1.640a140000000p-19f,0x1.6d5aa40000000p-19f,0x1.76d5200000000p-19f,
0x1.8079e00000000p-19f,0x1.8a49920000000p-19f,0x1.9443e00000000p-19f,0x1.9e69780000000p-19f,0x1.a8bab40000000p-19f,0x1.b3373c0000000p-19f,0x1.bde06c0000000p-19f,0x1.c8b59a0000000p-19f,
0x1.d3b7720000000p-19f,0x1.dee64c0000000p-19f,0x1.ea42280000000p-19f,0x1.f5cb620000000p-19f,0x1.00c1520000000p-18f,0x1.06b3a00000000p-18f,0x1.0cbd4e0000000p-18f,0x1.12ddfe0000000p-18f,
0x1.19160e0000000p-18f,0x1.1f657a0000000p-18f,0x1.25ccc80000000p-18f,0x1.2c4b9e0000000p-18f,0x1.32e2560000000p-18f,0x1.3990c40000000p-18f,0x1.40576c0000000p-18f,0x1.4736200000000p-18f,
0x1.4e2d0c0000000p-18f,0x1.553c5e0000000p-18f,0x1.5c64160000000p-18f,0x1.63a4600000000p-18f,0x1.6afd3a0000000p-18f,0x1.726efc0000000p-18f,0x1.79f97e0000000p-18f,0x1.819ce80000000p-18f,
0x1.8959660000000p-18f,0x1.912efa0000000p-18f,0x1.991dce0000000p-18f,0x1.a1258e0000000p-18f,0x1.a947100000000p-18f,0x1.b181d40000000p-18f,0x1.b9d6320000000p-18f,0x1.c244560000000p-18f,
0x1.cacc100000000p-18f,0x1.d36d660000000p-18f,0x1.dc28800000000p-18f,0x1.e4fd8a0000000p-18f,0x1.edecb20000000p-18f,0x1.f6f5a00000000p-18f,0x1.000c6a0000000p-17f,0x1.04aafe0000000p-17f,
0x1.0956e20000000p-17f,0x1.0e0fbe0000000p-17f,0x1.12d5c00000000p-17f,0x1.17a8fc0000000p-17f,0x1.1c895c0000000p-17f,0x1.2177220000000p-17f,0x1.2672220000000p-17f,0x1.2b7a5e0000000p-17f,
0x1.308fea0000000p-17f,0x1.35b2f20000000p-17f,0x1.3ae3200000000p-17f,0x1.4020e00000000p-17f,0x1.456bf00000000p-17f,0x1.4ac4500000000p-17f,0x1.502a420000000p-17f,0x1.559e200000000p-17f,
0x1.5b1ee00000000p-17f,0x1.60ad5e0000000p-17f,0x1.6648be0000000p-17f,0x1.6bf2ba0000000p-17f,0x1.71a9980000000p-17f,0x1.776e340000000p-17f,0x1.7d3fb20000000p-17f,0x1.831fcc0000000p-17f,
0x1.890cc60000000p-17f,0x1.8f07800000000p-17f,0x1.950ff80000000p-17f,0x1.9b26300000000p-17f,0x1.a14a260000000p-17f,0x1.a77afc0000000p-17f,0x1.adba6e0000000p-17f,0x1.b406c40000000p-17f,
0x1.ba60d60000000p-17f,0x1.c0c9840000000p-17f,0x1.c73f140000000p-17f,0x1.cdc2620000000p-17f,0x1.d453700000000p-17f,0x1.daf15e0000000p-17f,0x1.e19de80000000p-17f,0x1.e858300000000p-17f,
0x1.ef1f5c0000000p-17f,0x1.f5f5200000000p-17f,0x1.fcd7c80000000p-17f,0x1.01e4840000000p-16f,0x1.0563980000000p-16f,0x1.08e9880000000p-16f,0x1.0c76c80000000p-16f,0x1.100a780000000p-16f,
0x1.13a4980000000p-16f,0x1.1746060000000p-16f,0x1.1aee540000000p-16f,0x1.21ac040000000p-24f,0x1.34744a0000000p-24f,0x1.4833820000000p-24f,0x1.5cf27c0000000p-24f,0x1.72ba8c0000000p-24f,
0x1.8994d40000000p-24f,0x1.a18b000000000p-24f,0x1.baa6f80000000p-24f,0x1.d4f1140000000p-24f,0x1.f075a60000000p-24f,0x1.069ebc0000000p-23f,0x1.15a93a0000000p-23f,0x1.255fca0000000p-23f,
0x1.35c7ec0000000p-23f,0x1.46e7200000000p-23f,0x1.58c1cc0000000p-23f,0x1.6b5e880000000p-23f,0x1.7ec2460000000p-23f,0x1.92f2880000000p-23f,0x1.a7f5580000000p-23f,0x1.bdd0c20000000p-23f,
0x1.d48a460000000p-23f,0x1.ec27640000000p-23f,0x1.0257140000000p-22f,0x1.0f12940000000p-22f,0x1.1c48f40000000p-22f,0x1.29fcf00000000p-22f,0x1.3832200000000p-22f,0x1.46eb860000000p-22f,
0x1.562c2a0000000p-22f,0x1.65f7120000000p-22f,0x1.764f440000000p-22f,0x1.8738500000000p-22f,0x1.98b4fa0000000p-22f,0x1.aac8d20000000p-22f,0x1.bd76e00000000p-22f,0x1.d0c26e0000000p-22f,
0x1.e4ae840000000p-22f,0x1.f93efa0000000p-22f,0x1.073b240000000p-21f,0x1.122c4a0000000p-21f,0x1.1d74280000000p-21f,0x1.2914ce0000000p-21f,0x1.350f7c0000000p-21f,0x1.41661a0000000p-21f,
0x1.4e1a520000000p-21f,0x1.5b2dec0000000p-21f,0x1.68a2020000000p-21f,0x1.76792e0000000p-21f,0x1.84b4460000000p-21f,0x1.9355360000000p-21f,0x1.a25dea0000000p-21f,0x1.b1cfe60000000p-21f,
0x1.c1acac0000000p-21f,0x1.d1f64a0000000p-21f,0x1.e2ae000000000p-21f,0x1.f3d5de0000000p-21f,0x1.02b7a00000000p-20f,0x1.0bbdfa0000000p-20f,0x1.14fef00000000p-20f,0x1.1e7acc0000000p-20f,
0x1.2833220000000p-20f,0x1.3228a00000000p-20f,0x1.3c5bf80000000p-20f,0x1.46cd280000000p-20f,0x1.517e400000000p-20f,0x1.5c6ff20000000p-20f,0x1.67a23c0000000p-20f,0x1.73167e0000000p-20f,
0x1.7ece180000000p-20f,0x1.8ac85c0000000p-20f,0x1.9707580000000p-20f,0x1.a38bbc0000000p-20f,0x1.b054d80000000p-20f,0x1.bd661a0000000p-20f,0x1.cabd760000000p-20f,0x1.d85e5a0000000p-20f,
0x1.e647640000000p-20f,0x1.f47aa80000000p-20f,0x1.017c100000000p-19f,0x1.08e0f20000000p-19f,0x1.106b460000000p-19f,0x1.181c6e0000000p-19f,0x1.1ff46a0000000p-19f,0x1.27f33a0000000p-19f,
0x1.3019e40000000p-19f,0x1.38686a0000000p-19f,0x1.40df240000000p-19f,0x1.497e680000000p-19f,0x1.5246ea0000000p-19f,0x1.5b38a40000000p-19f,0x1.6453f40000000p-19f,0x1.6d99860000000p-19f,
0x1.77095a0000000p-19f,0x1.80a4240000000p-19f,0x1.8a69de0000000p-19f,0x1.945b3e0000000p-19f,0x1.9e77e80000000p-19f,0x1.a8c13c0000000p-19f,0x1.b3368c0000000p-19f,0x1.bdd8860000000p-19f,
0x1.c8a7dc0000000p-19f,0x1.d3a4340000000p-19f,0x1.dece960000000p-19f,0x1.ea27040000000p-19f,0x1.f5ad7c0000000p-19f,0x1.00b1580000000p-18f,0x1.06a37a0000000p-18f,0x1.0cad280000000p-18f,
0x1.12ce880000000p-18f,0x1.1907cc0000000p-18f,0x1.1f591c0000000p-18f,0x1.25c2a40000000p-18f,0x1.2c44940000000p-18f,0x1.32dee80000000p-18f,0x1.3991a00000000p-18f,0x1.405d420000000p-18f,
0x1.4741ce0000000p-18f,0x1.4e3f420000000p-18f,0x1.5555f80000000p-18f,0x1.5c85c20000000p-18f,0x1.63cefa0000000p-18f,0x1.6b31a00000000p-18f,0x1.72ae0a0000000p-18f,0x1.7a440c0000000p-18f,
0x1.81f42e0000000p-18f,0x1.89be120000000p-18f,0x1.91a1e80000000p-18f,0x1.99a0340000000p-18f,0x1.a1b89c0000000p-18f,0x1.a9eb4e0000000p-18f,0x1.b238f80000000p-18f,0x1.baa1180000000p-18f,
0x1.c324040000000p-18f,0x1.cbc1900000000p-18f,0x1.d47a160000000p-18f,0x1.dd4dc20000000p-18f,0x1.e63c920000000p-18f,0x1.ef46b20000000p-18f,0x1.f86bcc0000000p-18f,0x1.00d6480000000p-17f,
0x1.0584680000000p-17f,0x1.0a40460000000p-17f,0x1.0f0a0e0000000p-17f,0x1.13e1c20000000p-17f,0x1.18c7320000000p-17f,0x1.1dbaba0000000p-17f,0x1.22bc420000000p-17f,0x1.27cbe20000000p-17f,
0x1.2ce9540000000p-17f,0x1.3215200000000p-17f,0x1.374f020000000p-17f,0x1.3c97260000000p-17f,0x1.41ed760000000p-17f,0x1.4751f40000000p-17f,0x1.4cc4ca0000000p-17f,0x1.5245b60000000p-17f,
0x1.57d5ec0000000p-17f,0x1.5d73040000000p-17f,0x1.631f920000000p-17f,0x1.68dabc0000000p-17f,0x1.6ea3a40000000p-17f,0x1.747b260000000p-17f,0x1.7a61420000000p-17f,0x1.8055f80000000p-17f,
0x1.86594a0000000p-17f,0x1.8c6b340000000p-17f,0x1.928c960000000p-17f,0x1.98bbb60000000p-17f,0x1.9ef9700000000p-17f,0x1.a545c60000000p-17f,0x1.aba0b40000000p-17f,0x1.b20b1a0000000p-17f,
0x1.b884180000000p-17f,0x1.bf0ad60000000p-17f,0x1.c5a1e80000000p-17f,0x1.cc46b60000000p-17f,0x1.d2fa1e0000000p-17f,0x1.d9bcfe0000000p-17f,0x1.e08e780000000p-17f,0x1.e76f680000000p-17f,
0x1.c14a980000000p-26f,0x1.e216e80000000p-26f,0x1.026afa0000000p-25f,0x1.14cf860000000p-25f,0x1.2844180000000p-25f,0x1.3cd4fe0000000p-25f,0x1.528de20000000p-25f,0x1.697b4c0000000p-25f,
0x1.81aa340000000p-25f,0x1.9b275e0000000p-25f,0x1.b6006a0000000p-25f,0x1.d242c40000000p-25f,0x1.effcb40000000p-25f,0x1.079e440000000p-24f,0x1.1808440000000p-24f,0x1.29440e0000000p-24f,
0x1.3b591c0000000p-24f,0x1.4e4f3c0000000p-24f,0x1.622e740000000p-24f,0x1.76feb00000000p-24f,0x1.8cc8680000000p-24f,0x1.a393d80000000p-24f,0x1.bb69940000000p-24f,0x1.d451a60000000p-24f,
0x1.ee567e0000000p-24f,0x1.04bfe60000000p-23f,0x1.12ebbe0000000p-23f,0x1.21b3280000000p-23f,0x1.311a8e0000000p-23f,0x1.4126e20000000p-23f,0x1.51dda20000000p-23f,0x1.6343340000000p-23f,
0x1.755c8c0000000p-23f,0x1.882e9c0000000p-23f,0x1.9bbee20000000p-23f,0x1.b011c60000000p-23f,0x1.c52dde0000000p-23f,0x1.db17080000000p-23f,0x1.f1d34c0000000p-23f,0x1.04b4140000000p-22f,
0x1.10ed900000000p-22f,0x1.1d98da0000000p-22f,0x1.2ab8680000000p-22f,0x1.384f8a0000000p-22f,0x1.4660b60000000p-22f,0x1.54eef60000000p-22f,0x1.63fd4c0000000p-22f,0x1.738e7a0000000p-22f,
0x1.83a5860000000p-22f,0x1.9445740000000p-22f,0x1.a571080000000p-22f,0x1.b72b440000000p-22f,0x1.c977bc0000000p-22f,0x1.dc58ea0000000p-22f,0x1.efd2180000000p-22f,0x1.01f3280000000p-21f,
0x1.0c4c4a0000000p-21f,0x1.16f61a0000000p-21f,0x1.21f23e0000000p-21f,0x1.2d41f20000000p-21f,0x1.38e7460000000p-21f,0x1.44e3760000000p-21f,0x1.5138280000000p-21f,0x1.5de7260000000p-21f,
0x1.6af1f40000000p-21f,0x1.785a140000000p-21f,0x1.8621500000000p-21f,0x1.94492a0000000p-21f,0x1.a2d36e0000000p-21f,0x1.b1c1780000000p-21f,0x1.c115360000000p-21f,0x1.d0d0080000000p-21f,
0x1.e0f4000000000p-21f,0x1.f182580000000p-21f,0x1.013e6c0000000p-20f,0x1.09f2960000000p-20f,0x1.12deae0000000p-20f,0x1.1c03440000000p-20f,0x1.2560e40000000p-20f,0x1.2ef8ec0000000p-20f,
0x1.38cc0e0000000p-20f,0x1.42dbaa0000000p-20f,0x1.4d270e0000000p-20f,0x1.57b04c0000000p-20f,0x1.6278120000000p-20f,0x1.6d7f100000000p-20f,0x1.78c5f80000000p-20f,0x1.844e280000000p-20f,
0x1.9017a20000000p-20f,0x1.9c23c20000000p-20f,0x1.a8733c0000000p-20f,0x1.b506be0000000p-20f,0x1.c1dfa80000000p-20f,0x1.cefdfa0000000p-20f,0x1.dc63140000000p-20f,0x1.ea0ef60000000p-20f,
0x1.f803b00000000p-20f,0x1.0320f80000000p-19f,0x1.0a64de0000000p-19f,0x1.11cdde0000000p-19f,0x1.195caa0000000p-19f,0x1.21119a0000000p-19f,0x1.28ed5c0000000p-19f,0x1.30eff40000000p-19f,
0x1.3919b60000000p-19f,0x1.416afc0000000p-19f,0x1.49e4ce0000000p-19f,0x1.5286d20000000p-19f,0x1.5b51ba0000000p-19f,0x1.6446360000000p-19f,0x1.6d64440000000p-19f,0x1.76abe60000000p-19f,
0x1.801e7a0000000p-19f,0x1.89bbaa0000000p-19f,0x1.9384260000000p-19f,0x1.9d77ec0000000p-19f,0x1.a797ae0000000p-19f,0x1.b1e3c20000000p-19f,0x1.bc5c800000000p-19f,0x1.c7029a0000000p-19f,
0x1.d1d5b80000000p-19f,0x1.dcd6880000000p-19f,0x1.e805ba0000000p-19f,0x1.f362f80000000p-19f,0x1.feef480000000p-19f,0x1.0555540000000p-18f,0x1.0b4ae80000000p-18f,0x1.1158300000000p-18f,
0x1.177db00000000p-18f,0x1.1dbb6a0000000p-18f,0x1.2411b60000000p-18f,0x1.2a80660000000p-18f,0x1.3108000000000p-18f,0x1.37a8580000000p-18f,0x1.3e621c0000000p-18f,0x1.4534ca0000000p-18f,
0x1.4c21100000000p-18f,0x1.5326f00000000p-18f,0x1.5a46680000000p-18f,0x1.617fd20000000p-18f,0x1.68d32c0000000p-18f,0x1.7040fc0000000p-18f,0x1.77c8ea0000000p-18f,0x1.7f6b760000000p-18f,
0x1.87287a0000000p-18f,0x1.8f00760000000p-18f,0x1.96f33e0000000p-18f,0x1.9f01000000000p-18f,0x1.a72a3e0000000p-18f,0x1.af6ece0000000p-18f,0x1.b7ceae0000000p-18f,0x1.c04a360000000p-18f,
0x1.c8e1940000000p-18f,0x1.d194f40000000p-18f,0x1.da64260000000p-18f,0x1.e34fb40000000p-18f,0x1.ec576c0000000p-18f,0x1.f57b7e0000000p-18f,0x1.febc6c0000000p-18f,0x1.040cda0000000p-17f,
0x1.08ca020000000p-17f,0x1.0d95820000000p-17f,0x1.126fb40000000p-17f,0x1.1758680000000p-17f,0x1.1c4fb80000000p-17f,0x1.2155a20000000p-17f,0x1.266a680000000p-17f,0x1.2b8e0a0000000p-17f,
0x1.30c0720000000p-17f,0x1.3601ce0000000p-17f,0x1.3b521c0000000p-17f,0x1.40b1860000000p-17f,0x1.461fe40000000p-17f,0x1.4b9d8c0000000p-17f,0x1.512a3e0000000p-17f,0x1.56c67a0000000p-17f,
0x1.5c71520000000p-17f,0x1.622c7a0000000p-17f,0x1.67f63e0000000p-17f,0x1.6dcf780000000p-17f,0x1.73b8280000000p-17f,0x1.318fa80000000p-27f,0x1.4af9820000000p-27f,0x1.6626140000000p-27f,
0x1.832ec60000000p-27f,0x1.a22b180000000p-27f,0x1.c3372c0000000p-27f,0x1.e66e4a0000000p-27f,0x1.05f64a0000000p-26f,0x1.19e78c0000000p-26f,0x1.2f19da0000000p-26f,0x1.459d7e0000000p-26f,
0x1.5d824c0000000p-26f,0x1.76d8fa0000000p-26f,0x1.91b3200000000p-26f,0x1.ae21e40000000p-26f,0x1.cc38a20000000p-26f,0x1.ec08ee0000000p-26f,0x1.06d34c0000000p-25f,0x1.1893220000000p-25f,
0x1.2b4de20000000p-25f,0x1.3f0e1a0000000p-25f,0x1.53de8e0000000p-25f,0x1.69ca3e0000000p-25f,0x1.80dd0a0000000p-25f,0x1.9922280000000p-25f,0x1.b2a5b20000000p-25f,0x1.cd73c00000000p-25f,
0x1.e998da0000000p-25f,0x1.0390e00000000p-24f,0x1.130df00000000p-24f,0x1.234a160000000p-24f,0x1.344c440000000p-24f,0x1.461b480000000p-24f,0x1.58be800000000p-24f,0x1.6c3d120000000p-24f,
0x1.809e260000000p-24f,0x1.95e9be0000000p-24f,0x1.ac27580000000p-24f,0x1.c35efa0000000p-24f,0x1.db98380000000p-24f,0x1.f4da720000000p-24f,0x1.0797c80000000p-23f,0x1.154f9e0000000p-23f,
0x1.2399240000000p-23f,0x1.3278340000000p-23f,0x1.41f1320000000p-23f,0x1.5209100000000p-23f,0x1.62c4c20000000p-23f,0x1.7427960000000p-23f,0x1.8637940000000p-23f,0x1.98f80c0000000p-23f,
0x1.ac6f080000000p-23f,0x1.c0a0640000000p-23f,0x1.d5919c0000000p-23f,0x1.eb47a40000000p-23f,0x1.00e3b60000000p-22f,0x1.0c8b3e0000000p-22f,0x1.189c980000000p-22f,0x1.251a860000000p-22f,
0x1.3207c60000000p-22f,0x1.3f671c0000000p-22f,0x1.4d3afc0000000p-22f,0x1.5b86280000000p-22f,0x1.6a4b620000000p-22f,0x1.798d640000000p-22f,0x1.894f3a0000000p-22f,0x1.9993a00000000p-22f,
0x1.aa5d120000000p-22f,0x1.bbaeda0000000p-22f,0x1.cd8bb80000000p-22f,0x1.dff66c0000000p-22f,0x1.f2f2440000000p-22f,0x1.0340fe0000000p-21f,0x1.0d542c0000000p-21f,0x1.17b4760000000p-21f,
0x1.2263140000000p-21f,0x1.2d61f80000000p-21f,0x1.38b1f20000000p-21f,0x1.4455580000000p-21f,0x1.504d220000000p-21f,0x1.5c9b1a0000000p-21f,0x1.6940e20000000p-21f,0x1.7640020000000p-21f,
0x1.8399fa0000000p-21f,0x1.9150720000000p-21f,0x1.9f64ea0000000p-21f,0x1.add9300000000p-21f,0x1.bcaee60000000p-21f,0x1.cbe7040000000p-21f,0x1.db84040000000p-21f,0x1.eb87220000000p-21f,
0x1.fbf2040000000p-21f,0x1.0663280000000p-20f,0x1.0f03300000000p-20f,0x1.17d8be0000000p-20f,0x1.20e5f60000000p-20f,0x1.2a2b880000000p-20f,0x1.33a9740000000p-20f,0x1.3d611a0000000p-20f,
0x1.4753d80000000p-20f,0x1.5181000000000p-20f,0x1.5beb500000000p-20f,0x1.66921a0000000p-20f,0x1.7176be0000000p-20f,0x1.7c99e80000000p-20f,0x1.87fcfe0000000p-20f,0x1.939ffa0000000p-20f,
0x1.9f84400000000p-20f,0x1.abab2e0000000p-20f,0x1.b814160000000p-20f,0x1.c4c1b40000000p-20f,0x1.d1b35c0000000p-20f,0x1.deea6c0000000p-20f,0x1.ec67920000000p-20f,0x1.fa2c320000000p-20f,
0x1.041c240000000p-19f,0x1.0b469c0000000p-19f,0x1.1295d80000000p-19f,0x1.1a0a860000000p-19f,0x1.21a4a80000000p-19f,0x1.2964f00000000p-19f,0x1.314bb20000000p-19f,0x1.3958ee0000000p-19f,
0x1.418e080000000p-19f,0x1.49ea4c0000000p-19f,0x1.526f1c0000000p-19f,0x1.5b1c200000000p-19f,0x1.63f25e0000000p-19f,0x1.6cf1d80000000p-19f,0x1.761ae60000000p-19f,0x1.7f6e360000000p-19f,
0x1.88ebca0000000p-19f,0x1.9294aa0000000p-19f,0x1.9c68d40000000p-19f,0x1.a668f80000000p-19f,0x1.b095180000000p-19f,0x1.baed8a0000000p-19f,0x1.c573580000000p-19f,0x1.d026280000000p-19f,
0x1.db07040000000p-19f,0x1.e615920000000p-19f,0x1.f152dc0000000p-19f,0x1.fcbf360000000p-19f,0x1.042d7e0000000p-18f,0x1.0a12ea0000000p-18f,0x1.1010bc0000000p-18f,0x1.16269c0000000p-18f,
0x1.1c54e00000000p-18f,0x1.229bb40000000p-18f,0x1.28fb740000000p-18f,0x1.2f73ee0000000p-18f,0x1.3605ac0000000p-18f,0x1.3cb0aa0000000p-18f,0x1.43756c0000000p-18f,0x1.4a53c80000000p-18f,
0x1.514bea0000000p-18f,0x1.585e280000000p-18f,0x1.5f8ada0000000p-18f,0x1.66d1800000000p-18f,0x1.6e33480000000p-18f,0x1.75af860000000p-18f,0x1.7d46be0000000p-18f,0x1.84f8ee0000000p-18f,
0x1.8cc69a0000000p-18f,0x1.94afc40000000p-18f,0x1.9cb46a0000000p-18f,0x1.a4d4e60000000p-18f,0x1.ad11360000000p-18f,0x1.b56a0c0000000p-18f,0x1.bddeb60000000p-18f,0x1.c670100000000p-18f,
0x1.cf1e1a0000000p-18f,0x1.d7e8800000000p-18f,0x1.e0d0440000000p-18f,0x1.e9d4e60000000p-18f,0x1.f2f6bc0000000p-18f,0x1.fc361e0000000p-18f,0x1.02c95a0000000p-17f,0x1.0786ae0000000p-17f,
0x1.0c52ca0000000p-17f,0x1.112dec0000000p-17f,0x1.5cc6540000000p-29f,0x1.7d09dc0000000p-29f,0x1.9fd73e0000000p-29f,0x1.c558000000000p-29f,0x1.edb81e0000000p-29f,0x1.0c92aa0000000p-28f,
0x1.23e7900000000p-28f,0x1.3cf3ea0000000p-28f,0x1.57d1940000000p-28f,0x1.749c2c0000000p-28f,0x1.9370340000000p-28f,0x1.b46b0c0000000p-28f,0x1.d7ab7c0000000p-28f,0x1.fd52120000000p-28f,
0x1.12c0060000000p-27f,0x1.282c080000000p-27f,0x1.3eff480000000p-27f,0x1.574c680000000p-27f,0x1.7126960000000p-27f,0x1.8ca2940000000p-27f,0x1.a9d49e0000000p-27f,0x1.c8d2b20000000p-27f,
0x1.e9b3ae0000000p-27f,0x1.0646c60000000p-26f,0x1.18bc7a0000000p-26f,0x1.2c47b20000000p-26f,0x1.40f44e0000000p-26f,0x1.56cf800000000p-26f,0x1.6de7cc0000000p-26f,0x1.864a620000000p-26f,
0x1.a006360000000p-26f,0x1.bb295e0000000p-26f,0x1.d7c41e0000000p-26f,0x1.f5e4f80000000p-26f,0x1.0ace8a0000000p-25f,0x1.1b7e260000000p-25f,0x1.2d0a500000000p-25f,0x1.3f7b2c0000000p-25f,
0x1.52da300000000p-25f,0x1.672ff00000000p-25f,0x1.7c86c00000000p-25f,0x1.92e7a60000000p-25f,0x1.aa5cf60000000p-25f,0x1.c2f0260000000p-25f,0x1.dcac6c0000000p-25f,0x1.f79bac0000000p-25f,
0x1.09e4ae0000000p-24f,0x1.18a0000000000p-24f,0x1.2805c00000000p-24f,0x1.381b6c0000000p-24f,0x1.48e7480000000p-24f,0x1.5a6f0c0000000p-24f,0x1.6cb91a0000000p-24f,0x1.7fcb5e0000000p-24f,
0x1.93acca0000000p-24f,0x1.a863680000000p-24f,0x1.bdf6b00000000p-24f,0x1.d46c0a0000000p-24f,0x1.ebca7a0000000p-24f,0x1.020d680000000p-23f,0x1.0eb1460000000p-23f,0x1.1bd4ae0000000p-23f,
0x1.297aec0000000p-23f,0x1.37a8f60000000p-23f,0x1.4660fc0000000p-23f,0x1.55a7f40000000p-23f,0x1.6581b20000000p-23f,0x1.75f2140000000p-23f,0x1.86fcf20000000p-23f,0x1.98a6220000000p-23f,
0x1.aaf29a0000000p-23f,0x1.bde6320000000p-23f,0x1.d185520000000p-23f,0x1.e5d3ce0000000p-23f,0x1.fad69e0000000p-23f,0x1.0849120000000p-22f,0x1.1385660000000p-22f,0x1.1f227a0000000p-22f,
0x1.2b22ca0000000p-22f,0x1.3788400000000p-22f,0x1.4455e60000000p-22f,0x1.518da40000000p-22f,0x1.5f323e0000000p-22f,0x1.6d459e0000000p-22f,0x1.7bcb100000000p-22f,0x1.8ac43c0000000p-22f,
0x1.9a33e20000000p-22f,0x1.aa1cc00000000p-22f,0x1.ba81960000000p-22f,0x1.cb64500000000p-22f,0x1.dcc8820000000p-22f,0x1.eeafd40000000p-22f,0x1.008ec60000000p-21f,0x1.0a0a160000000p-21f,
0x1.13cb380000000p-21f,0x1.1dd3660000000p-21f,0x1.28246e0000000p-21f,0x1.32bf1e0000000p-21f,0x1.3da5680000000p-21f,0x1.48d83e0000000p-21f,0x1.54596c0000000p-21f,0x1.602a2e0000000p-21f,
0x1.6c4c060000000p-21f,0x1.78c0320000000p-21f,0x1.85887c0000000p-21f,0x1.92a61e0000000p-21f,0x1.a01ac00000000p-21f,0x1.ade79e0000000p-21f,0x1.bc0e820000000p-21f,0x1.ca90cc0000000p-21f,
0x1.d96fda0000000p-21f,0x1.e8ad780000000p-21f,0x1.f84ae20000000p-21f,0x1.0424f00000000p-20f,0x1.0c55d80000000p-20f,0x1.14b8d80000000p-20f,0x1.1d4f2e0000000p-20f,0x1.2619be0000000p-20f,
0x1.2f188a0000000p-20f,0x1.384c3e0000000p-20f,0x1.41b6ec0000000p-20f,0x1.4b57e20000000p-20f,0x1.5531340000000p-20f,0x1.5f42e00000000p-20f,0x1.698e440000000p-20f,0x1.7413620000000p-20f,
0x1.7ed3980000000p-20f,0x1.89cf980000000p-20f,0x1.9508120000000p-20f,0x1.a07db40000000p-20f,0x1.ac31e00000000p-20f,0x1.b825440000000p-20f,0x1.c458900000000p-20f,0x1.d0cbc60000000p-20f,
0x1.dd80f40000000p-20f,0x1.ea78ca0000000p-20f,0x1.f7b34a0000000p-20f,0x1.0298e80000000p-19f,0x1.097a880000000p-19f,0x1.107f340000000p-19f,0x1.17a6940000000p-19f,0x1.1ef2080000000p-19f,
0x1.2661360000000p-19f,0x1.2df4d20000000p-19f,0x1.35ad8a0000000p-19f,0x1.3d8b5c0000000p-19f,0x1.458efc0000000p-19f,0x1.4db8660000000p-19f,0x1.5608a40000000p-19f,0x1.5e7fb60000000p-19f,
0x1.671df20000000p-19f,0x1.6fe40a0000000p-19f,0x1.78d2580000000p-19f,0x1.81e92e0000000p-19f,0x1.8b28ea0000000p-19f,0x1.9492380000000p-19f,0x1.9e25180000000p-19f,0x1.a7e2960000000p-19f,
0x1.b1ca540000000p-19f,0x1.bbdd080000000p-19f,0x1.c61b5e0000000p-19f,0x1.d085ae0000000p-19f,0x1.db1bfa0000000p-19f,0x1.e5def00000000p-19f,0x1.f0ceea0000000p-19f,0x1.fbec960000000p-19f,
0x1.039bfa0000000p-18f,0x1.0959080000000p-18f,0x1.0f2d1a0000000p-18f,0x1.1518e20000000p-18f,0x1.1b1c5e0000000p-18f,0x1.2137be0000000p-18f,0x1.276b540000000p-18f,0x1.2db7520000000p-18f,
0x1.341bb40000000p-18f,0x1.3a98d20000000p-18f,0x1.412eda0000000p-18f,0x1.47de240000000p-18f,0x1.4ea6ae0000000p-18f,0x1.5588d20000000p-18f,0x1.5c848e0000000p-18f,0x1.2d2de80000000p-32f,
0x1.4ecd220000000p-32f,0x1.73b24e0000000p-32f,0x1.9c22ac0000000p-32f,0x1.c867fc0000000p-32f,0x1.f8d07a0000000p-32f,0x1.16d8060000000p-31f,0x1.33afd20000000p-31f,0x1.531ee80000000p-31f,
0x1.7557400000000p-31f,0x1.9a8e320000000p-31f,0x1.c2fbe40000000p-31f,0x1.eedc020000000p-31f,0x1.0f36cc0000000p-30f,0x1.28f9b00000000p-30f,0x1.44d9c80000000p-30f,0x1.62fb9a0000000p-30f,
0x1.8387440000000p-30f,0x1.a6a4de0000000p-30f,0x1.cc800e0000000p-30f,0x1.f5468e0000000p-30f,0x1.10941c0000000p-29f,0x1.282b820000000p-29f,0x1.4183820000000p-29f,0x1.5cb7e60000000p-29f,
0x1.79e5e40000000p-29f,0x1.992b640000000p-29f,0x1.baa8100000000p-29f,0x1.de7ca20000000p-29f,0x1.0266240000000p-28f,0x1.16dd440000000p-28f,0x1.2cb6860000000p-28f,0x1.4405720000000p-28f,
0x1.5cde9a0000000p-28f,0x1.77574a0000000p-28f,0x1.93857e0000000p-28f,0x1.b180420000000p-28f,0x1.d15f2a0000000p-28f,0x1.f33bb60000000p-28f,0x1.0b97720000000p-27f,0x1.1eaa280000000p-27f,
0x1.32e3720000000p-27f,0x1.4851ec0000000p-27f,0x1.5f04020000000p-27f,0x1.770a3c0000000p-27f,0x1.9073d40000000p-27f,0x1.ab509c0000000p-27f,0x1.c7b2300000000p-27f,0x1.e5ab040000000p-27f,
0x1.02a65a0000000p-26f,0x1.1354da0000000p-26f,0x1.24eab20000000p-26f,0x1.3771fc0000000p-26f,0x1.4af4d80000000p-26f,0x1.5f7e460000000p-26f,0x1.7518d00000000p-26f,0x1.8bd05a0000000p-26f,
0x1.a3af700000000p-26f,0x1.bcc2d40000000p-26f,0x1.d716640000000p-26f,0x1.f2b6e40000000p-26f,0x1.07d88a0000000p-25f,0x1.17094a0000000p-25f,0x1.26f4140000000p-25f,0x1.37a09c0000000p-25f,
0x1.49153e0000000p-25f,0x1.5b5a200000000p-25f,0x1.6e76120000000p-25f,0x1.8271380000000p-25f,0x1.97537c0000000p-25f,0x1.ad24c80000000p-25f,0x1.c3ed780000000p-25f,0x1.dbb6200000000p-25f,
0x1.f487540000000p-25f,0x1.0734ee0000000p-24f,0x1.14b37c0000000p-24f,0x1.22c3f20000000p-24f,0x1.316ad20000000p-24f,0x1.40ad460000000p-24f,0x1.50900a0000000p-24f,0x1.61182c0000000p-24f,
0x1.724ad60000000p-24f,0x1.842d500000000p-24f,0x1.96c4a80000000p-24f,0x1.aa16d00000000p-24f,0x1.be282c0000000p-24f,0x1.d3001a0000000p-24f,0x1.e8a3700000000p-24f,0x1.ff17b00000000p-24f,
0x1.0b31b80000000p-23f,0x1.1746a40000000p-23f,0x1.23ccd20000000p-23f,0x1.30c78c0000000p-23f,0x1.3e3aaa0000000p-23f,0x1.4c297c0000000p-23f,0x1.5a96300000000p-23f,0x1.6985320000000p-23f,
0x1.78f9c80000000p-23f,0x1.88f6b80000000p-23f,0x1.997fd60000000p-23f,0x1.aa98fe0000000p-23f,0x1.bc457c0000000p-23f,0x1.ce89280000000p-23f,0x1.e167dc0000000p-23f,0x1.f4e4e40000000p-23f,
0x1.04820e0000000p-22f,0x1.0ee4ac0000000p-22f,0x1.199cc80000000p-22f,0x1.24abbe0000000p-22f,0x1.3013800000000p-22f,0x1.3bd6820000000p-22f,0x1.47f6b60000000p-22f,0x1.54764a0000000p-22f,
0x1.61572c0000000p-22f,0x1.6e9b900000000p-22f,0x1.7c45620000000p-22f,0x1.8a57620000000p-22f,0x1.98d2f00000000p-22f,0x1.a7bb580000000p-22f,0x1.b711fa0000000p-22f,0x1.c6d94e0000000p-22f,
0x1.d713d00000000p-22f,0x1.e7c3f60000000p-22f,0x1.f8ebf60000000p-22f,0x1.0546de0000000p-21f,0x1.0e56260000000p-21f,0x1.17a5100000000p-21f,0x1.2134940000000p-21f,0x1.2b06100000000p-21f,
0x1.351a9e0000000p-21f,0x1.3f73c00000000p-21f,0x1.4a12920000000p-21f,0x1.54f8720000000p-21f,0x1.60269c0000000p-21f,0x1.6b9e700000000p-21f,0x1.7760e80000000p-21f,0x1.836fa60000000p-21f,
0x1.8fcc2e0000000p-21f,0x1.9c77540000000p-21f,0x1.a972e00000000p-21f,0x1.b6bfec0000000p-21f,0x1.c45ffc0000000p-21f,0x1.d2544c0000000p-21f,0x1.e09e840000000p-21f,0x1.ef3fba0000000p-21f,
0x1.fe394e0000000p-21f,0x1.06c6860000000p-20f,0x1.0e9e2a0000000p-20f,0x1.16a4300000000p-20f,0x1.1ed8f20000000p-20f,0x1.273dce0000000p-20f,0x1.2fd3740000000p-20f,0x1.3899e40000000p-20f,
0x1.4193300000000p-20f,0x1.4abea40000000p-20f,0x1.541da40000000p-20f,0x1.5db0dc0000000p-20f,0x1.6779b00000000p-20f,0x1.71776e0000000p-20f,0x1.7bacd40000000p-20f,0x1.8618840000000p-20f,
0x1.90bc8e0000000p-20f,0x1.9b99a20000000p-20f,0x1.a6b0700000000p-20f,0x1.b200f60000000p-20f,0x1.bd8d460000000p-20f,0x1.c955600000000p-20f,0x1.d559f20000000p-20f,0x1.e19c5c0000000p-20f,
0x1.ee1ca00000000p-20f,0x1.fadcce0000000p-20f,0x1.03ee1a0000000p-19f,0x1.0a8e180000000p-19f,0x1.114f140000000p-19f,0x1.18310c0000000p-19f,0x1.1f34b00000000p-19f,0x1.265a580000000p-19f,
0x1.2da2040000000p-19f,0x1.350cbc0000000p-19f,0x1.3c9a800000000p-19f,0x1.444ba80000000p-19f,0x1.8f054c0000000p-35f,0x1.c1dcec0000000p-35f,0x1.fa70460000000p-35f,0x1.1ca6b80000000p-34f,
0x1.3f881e0000000p-34f,0x1.662f8a0000000p-34f,0x1.90f72c0000000p-34f,0x1.c040d80000000p-34f,0x1.f474260000000p-34f,0x1.1701660000000p-33f,0x1.36b2120000000p-33f,0x1.598df40000000p-33f,
0x1.7fd9ba0000000p-33f,0x1.a9dfb40000000p-33f,0x1.d7f0600000000p-33f,0x1.052fd20000000p-32f,0x1.20c49e0000000p-32f,0x1.3ee6b80000000p-32f,0x1.5fc95c0000000p-32f,0x1.83a4420000000p-32f,
0x1.aab0d60000000p-32f,0x1.d52de00000000p-32f,0x1.01ae560000000p-31f,0x1.1ac1a80000000p-31f,0x1.35f5e80000000p-31f,0x1.5372140000000p-31f,0x1.735fd60000000p-31f,0x1.95eaa60000000p-31f,
0x1.bb41660000000p-31f,0x1.e394360000000p-31f,0x1.078b700000000p-30f,0x1.1f00020000000p-30f,0x1.38444c0000000p-30f,0x1.5375e20000000p-30f,0x1.70b51e0000000p-30f,0x1.9022600000000p-30f,
0x1.b1e18c0000000p-30f,0x1.d616840000000p-30f,0x1.fce7fe0000000p-30f,0x1.133f0a0000000p-29f,0x1.2981800000000p-29f,0x1.4151320000000p-29f,0x1.5ac5560000000p-29f,0x1.75f6360000000p-29f,
0x1.92fcce0000000p-29f,0x1.b1f3220000000p-29f,0x1.d2f55e0000000p-29f,0x1.f61ef00000000p-29f,0x1.0dc73a0000000p-28f,0x1.21b1720000000p-28f,0x1.36de200000000p-28f,0x1.4d5e560000000p-28f,
0x1.6543500000000p-28f,0x1.7e9f2e0000000p-28f,0x1.9984c40000000p-28f,0x1.b607f00000000p-28f,0x1.d43cc20000000p-28f,0x1.f438840000000p-28f,0x1.0b086c0000000p-27f,0x1.1cee220000000p-27f,
0x1.2fd9580000000p-27f,0x1.43d6080000000p-27f,0x1.58f0dc0000000p-27f,0x1.6f367c0000000p-27f,0x1.86b4600000000p-27f,0x1.9f797a0000000p-27f,0x1.b992fe0000000p-27f,0x1.d50efc0000000p-27f,
0x1.f1fe2c0000000p-27f,0x1.0838320000000p-26f,0x1.183a4c0000000p-26f,0x1.290e2e0000000p-26f,0x1.3abca60000000p-26f,0x1.4d4d9e0000000p-26f,0x1.60cba40000000p-26f,0x1.753f140000000p-26f,
0x1.8ab19a0000000p-26f,0x1.a12dc40000000p-26f,0x1.b8bdb00000000p-26f,0x1.d16b7a0000000p-26f,0x1.eb42240000000p-26f,0x1.03261c0000000p-25f,0x1.114b060000000p-25f,0x1.20154e0000000p-25f,
0x1.2f8b540000000p-25f,0x1.3fb2960000000p-25f,0x1.5091b00000000p-25f,0x1.622f3a0000000p-25f,0x1.7491940000000p-25f,0x1.87bfc60000000p-25f,0x1.9bc06c0000000p-25f,0x1.b09a8c0000000p-25f,
0x1.c655a40000000p-25f,0x1.dcf92a0000000p-25f,0x1.f48c620000000p-25f,0x1.068b9c0000000p-24f,0x1.1350c80000000p-24f,0x1.2099cc0000000p-24f,0x1.2e6ab40000000p-24f,0x1.3cc7940000000p-24f,
0x1.4bb4ee0000000p-24f,0x1.5b37280000000p-24f,0x1.6b52a80000000p-24f,0x1.7c0c0c0000000p-24f,0x1.8d67b80000000p-24f,0x1.9f6abe0000000p-24f,0x1.b219d60000000p-24f,0x1.c579bc0000000p-24f,
0x1.d990260000000p-24f,0x1.ee617c0000000p-24f,0x1.01f9120000000p-23f,0x1.0d24e60000000p-23f,0x1.18b5e20000000p-23f,0x1.24aec60000000p-23f,0x1.3112dc0000000p-23f,0x1.3de4e80000000p-23f,
0x1.4b27a60000000p-23f,0x1.58de620000000p-23f,0x1.670ac60000000p-23f,0x1.75b1c20000000p-23f,0x1.84d4fc0000000p-23f,0x1.9477c20000000p-23f,0x1.a49dec0000000p-23f,0x1.b549ae0000000p-23f,
0x1.c67ff80000000p-23f,0x1.d841e60000000p-23f,0x1.ea93de0000000p-23f,0x1.fd7a440000000p-23f,0x1.087b600000000p-22f,0x1.1286dc0000000p-22f,0x1.1ce1820000000p-22f,0x1.278cb20000000p-22f,
0x1.328a9e0000000p-22f,0x1.3ddcee0000000p-22f,0x1.49858e0000000p-22f,0x1.55866a0000000p-22f,0x1.61e12a0000000p-22f,0x1.6e97fe0000000p-22f,0x1.7bacd40000000p-22f,0x1.8921e00000000p-22f,
0x1.96f8800000000p-22f,0x1.a533740000000p-22f,0x1.b3d41e0000000p-22f,0x1.c2dcf60000000p-22f,0x1.d2502e0000000p-22f,0x1.e22fb20000000p-22f,0x1.f27d720000000p-22f,0x1.019e140000000p-21f,
0x1.0a36e40000000p-21f,0x1.1309fa0000000p-21f,0x1.1c18d80000000p-21f,0x1.2564760000000p-21f,0x1.2eedee0000000p-21f,0x1.38b67a0000000p-21f,0x1.42bf5a0000000p-21f,0x1.4d095e0000000p-21f,
0x1.57960c0000000p-21f,0x1.62667a0000000p-21f,0x1.6d7be80000000p-21f,0x1.78d76c0000000p-21f,0x1.847a460000000p-21f,0x1.9065f80000000p-21f,0x1.9c9b320000000p-21f,0x1.a91bbc0000000p-21f,
0x1.b5e8b20000000p-21f,0x1.c3032a0000000p-21f,0x1.d06caa0000000p-21f,0x1.de264c0000000p-21f,0x1.ec31b20000000p-21f,0x1.fa8fb40000000p-21f,0x1.04a0e80000000p-20f,0x1.0c24b60000000p-20f,
0x1.13d4020000000p-20f,0x1.1baf6e0000000p-20f,0x1.23b7840000000p-20f,0x1.2becf60000000p-20f,0x1.3450720000000p-20f,0x1.3ce3580000000p-20f,0x1.45a4f80000000p-20f,0x1.4e97640000000p-20f,
0x1.57ba9a0000000p-20f,0x1.ecfa920000000p-37f,0x1.1858760000000p-36f,0x1.3e59240000000p-36f,0x1.68f0ce0000000p-36f,0x1.989b4e0000000p-36f,0x1.cde0340000000p-36f,0x1.04a8b60000000p-35f,
0x1.25c7dc0000000p-35f,0x1.4aa4420000000p-35f,0x1.739b900000000p-35f,0x1.a1143e0000000p-35f,0x1.d37b820000000p-35f,0x1.05a45a0000000p-34f,0x1.247e6c0000000p-34f,0x1.46908c0000000p-34f,
0x1.6c25b20000000p-34f,0x1.958db60000000p-34f,0x1.c31eee0000000p-34f,0x1.f534ae0000000p-34f,0x1.161a080000000p-33f,0x1.3442e40000000p-33f,0x1.554d360000000p-33f,0x1.7976ec0000000p-33f,
0x1.a100340000000p-33f,0x1.cc2db80000000p-33f,0x1.fb4a5c0000000p-33f,0x1.1751e80000000p-32f,0x1.3346f80000000p-32f,0x1.51b0d80000000p-32f,0x1.72be8c0000000p-32f,0x1.96a2bc0000000p-32f,
0x1.bd91c40000000p-32f,0x1.e7c4c60000000p-32f,0x1.0abb700000000p-31f,0x1.2373d60000000p-31f,0x1.3e2c860000000p-31f,0x1.5b093e0000000p-31f,0x1.7a2eba0000000p-31f,0x1.9bc4aa0000000p-31f,
0x1.bff4480000000p-31f,0x1.e6e95e0000000p-31f,0x1.0868ae0000000p-30f,0x1.1eee6a0000000p-30f,0x1.371f920000000p-30f,0x1.5115e60000000p-30f,0x1.6cee400000000p-30f,0x1.8ac57a0000000p-30f,
0x1.aaba8c0000000p-30f,0x1.ccee880000000p-30f,0x1.f181cc0000000p-30f,0x1.0c4c1e0000000p-29f,0x1.212b920000000p-29f,0x1.3772400000000p-29f,0x1.4f34ec0000000p-29f,0x1.6888000000000p-29f,
0x1.8381fc0000000p-29f,0x1.a039c20000000p-29f,0x1.bec6e40000000p-29f,0x1.df425e0000000p-29f,0x1.00e31c0000000p-28f,0x1.1336f20000000p-28f,0x1.26aa3c0000000p-28f,0x1.3b4c4a0000000p-28f,
0x1.512b880000000p-28f,0x1.6858520000000p-28f,0x1.80e27e0000000p-28f,0x1.9adb480000000p-28f,0x1.b654760000000p-28f,0x1.d35ffa0000000p-28f,0x1.f211000000000p-28f,0x1.093d700000000p-27f,
0x1.1a59040000000p-27f,0x1.2c65b00000000p-27f,0x1.3f6e4a0000000p-27f,0x1.537e260000000p-27f,0x1.68a0460000000p-27f,0x1.7ee2620000000p-27f,0x1.964eae0000000p-27f,0x1.aef25c0000000p-27f,
0x1.c8dc600000000p-27f,0x1.e418280000000p-27f,0x1.005a540000000p-26f,0x1.0f5ffc0000000p-26f,0x1.1f24f40000000p-26f,0x1.2fb0b60000000p-26f,0x1.410a4c0000000p-26f,0x1.533af20000000p-26f,
0x1.664a940000000p-26f,0x1.7a41fe0000000p-26f,0x1.8f29180000000p-26f,0x1.a50a020000000p-26f,0x1.bbedf60000000p-26f,0x1.d3ddc20000000p-26f,0x1.ece3f40000000p-26f,0x1.0385540000000p-25f,
0x1.112d8e0000000p-25f,0x1.1f70980000000p-25f,0x1.2e53800000000p-25f,0x1.3ddc000000000p-25f,0x1.4e0fcc0000000p-25f,0x1.5ef50e0000000p-25f,0x1.70910e0000000p-25f,0x1.82ea620000000p-25f,
0x1.9606fe0000000p-25f,0x1.a9ede80000000p-25f,0x1.bea5120000000p-25f,0x1.d433860000000p-25f,0x1.ea9fa20000000p-25f,0x1.00f8c60000000p-24f,0x1.0d17980000000p-24f,0x1.19b03e0000000p-24f,
0x1.26c6740000000p-24f,0x1.345dc00000000p-24f,0x1.427a860000000p-24f,0x1.51204c0000000p-24f,0x1.6053780000000p-24f,0x1.7017fe0000000p-24f,0x1.8071d60000000p-24f,0x1.91660c0000000p-24f,
0x1.a2f80a0000000p-24f,0x1.b52d160000000p-24f,0x1.c8090c0000000p-24f,0x1.db91a00000000p-24f,0x1.efc9060000000p-24f,0x1.025b7a0000000p-23f,0x1.0d2fe40000000p-23f,0x1.18636a0000000p-23f,
0x1.23f95a0000000p-23f,0x1.2ff3e40000000p-23f,0x1.3c553a0000000p-23f,0x1.49213a0000000p-23f,0x1.5659860000000p-23f,0x1.64016c0000000p-23f,0x1.721baa0000000p-23f,0x1.80aa760000000p-23f,
0x1.8fb1a60000000p-23f,0x1.9f33700000000p-23f,0x1.af32920000000p-23f,0x1.bfb2e40000000p-23f,0x1.d0b7280000000p-23f,0x1.e241900000000p-23f,0x1.f456820000000p-23f,0x1.037ca40000000p-22f,
0x1.0d15c80000000p-22f,0x1.16f8da0000000p-22f,0x1.2127cc0000000p-22f,0x1.2ba36e0000000p-22f,0x1.366e3a0000000p-22f,0x1.4189900000000p-22f,0x1.4cf75e0000000p-22f,0x1.58b9460000000p-22f,
0x1.64d0f40000000p-22f,0x1.7140960000000p-22f,0x1.7e09d60000000p-22f,0x1.8b2e560000000p-22f,0x1.98b04e0000000p-22f,0x1.a691600000000p-22f,0x1.b4d37a0000000p-22f,0x1.c3788a0000000p-22f,
0x1.d283080000000p-22f,0x1.e1f49a0000000p-22f,0x1.f1cee60000000p-22f,0x1.010a320000000p-21f,0x1.0963820000000p-21f,0x1.11f45a0000000p-21f,0x1.1abdf40000000p-21f,0x1.23c1460000000p-21f,
0x1.2cff6c0000000p-21f,0x1.3679380000000p-21f,0x1.40302e0000000p-21f,0x1.4a25200000000p-21f,0x1.54594a0000000p-21f,0x1.5ecd800000000p-21f,0x1.6983440000000p-21f,0x1.747b8e0000000p-21f,
0x1.7fb7e20000000p-21f,0x1.8b38a60000000p-21f,0x1.96ffc80000000p-21f,0x1.a30dd60000000p-21f,0x1.af64bc0000000p-21f,0x1.bc04e20000000p-21f,0x1.12c72a0000000p-38f,0x1.3b40540000000p-38f,
0x1.6913d80000000p-38f,0x1.9ce1800000000p-38f,0x1.d759aa0000000p-38f,0x1.0c9efe0000000p-37f,0x1.31b1840000000p-37f,0x1.5b58f60000000p-37f,0x1.8a15220000000p-37f,0x1.be6fda0000000p-37f,
0x1.f903f00000000p-37f,0x1.1d3aaa0000000p-36f,0x1.41bd6e0000000p-36f,0x1.6a6b8e0000000p-36f,0x1.97b0ac0000000p-36f,0x1.ca00100000000p-36f,0x1.00ec260000000p-35f,0x1.1fe03a0000000p-35f,
0x1.42265e0000000p-35f,0x1.680dd20000000p-35f,0x1.91ec9e0000000p-35f,0x1.c01f120000000p-35f,0x1.f3092a0000000p-35f,0x1.158ab40000000p-34f,0x1.345be20000000p-34f,0x1.5635d20000000p-34f,
0x1.7b5af20000000p-34f,0x1.a412140000000p-34f,0x1.d0a8740000000p-34f,0x1.00b6c80000000p-33f,0x1.1b5c9a0000000p-33f,0x1.3874fe0000000p-33f,0x1.58305e0000000p-33f,0x1.7ac4300000000p-33f,
0x1.a068c40000000p-33f,0x1.c959340000000p-33f,0x1.f5d63e0000000p-33f,0x1.1311700000000p-32f,0x1.2d43dc0000000p-32f,0x1.49a7d40000000p-32f,0x1.68666e0000000p-32f,0x1.89a99c0000000p-32f,
0x1.ad9f400000000p-32f,0x1.d477320000000p-32f,0x1.fe638e0000000p-32f,0x1.15cd0c0000000p-31f,0x1.2e29b00000000p-31f,0x1.48651e0000000p-31f,0x1.649edc0000000p-31f,0x1.82f7da0000000p-31f,
0x1.a392fe0000000p-31f,0x1.c694740000000p-31f,0x1.ec23120000000p-31f,0x1.0a33340000000p-30f,0x1.1fc50a0000000p-30f,0x1.36dcb80000000p-30f,0x1.4f91760000000p-30f,0x1.69fbe60000000p-30f,
0x1.86355c0000000p-30f,0x1.a4594a0000000p-30f,0x1.c483240000000p-30f,0x1.e6cfc00000000p-30f,0x1.05af660000000p-29f,0x1.19279e0000000p-29f,0x1.2de16a0000000p-29f,0x1.43ee0c0000000p-29f,
0x1.5b60260000000p-29f,0x1.7449aa0000000p-29f,0x1.8ebf5c0000000p-29f,0x1.aad4960000000p-29f,0x1.c89fda0000000p-29f,0x1.e836a20000000p-29f,0x1.04d8120000000p-28f,0x1.1692520000000p-28f,
0x1.2956380000000p-28f,0x1.3d30f40000000p-28f,0x1.5230120000000p-28f,0x1.6860f20000000p-28f,0x1.7fd2b20000000p-28f,0x1.9893c00000000p-28f,0x1.b2b3c60000000p-28f,0x1.ce42c40000000p-28f,
0x1.eb51cc0000000p-28f,0x1.04f89e0000000p-27f,0x1.1519da0000000p-27f,0x1.2615800000000p-27f,0x1.37f5220000000p-27f,0x1.4ac2880000000p-27f,0x1.5e87fc0000000p-27f,0x1.734fdc0000000p-27f,
0x1.8924e60000000p-27f,0x1.a011a60000000p-27f,0x1.b823500000000p-27f,0x1.d163900000000p-27f,0x1.ebe0780000000p-27f,0x1.03d2bc0000000p-26f,0x1.1260520000000p-26f,0x1.219eb60000000p-26f,
0x1.3195600000000p-26f,0x1.424aea0000000p-26f,0x1.53c73e0000000p-26f,0x1.6611680000000p-26f,0x1.7931500000000p-26f,0x1.8d2e700000000p-26f,0x1.a211960000000p-26f,0x1.b7e1c80000000p-26f,
0x1.cea8460000000p-26f,0x1.e66e4a0000000p-26f,0x1.ff3c300000000p-26f,0x1.0c8d620000000p-25f,0x1.1a0a120000000p-25f,0x1.2818c60000000p-25f,0x1.36bec40000000p-25f,0x1.4600740000000p-25f,
0x1.55e3c40000000p-25f,0x1.666d8c0000000p-25f,0x1.77a34a0000000p-25f,0x1.898ab60000000p-25f,0x1.9c29500000000p-25f,0x1.af85080000000p-25f,0x1.c3a3960000000p-25f,0x1.d88b5a0000000p-25f,
0x1.ee42420000000p-25f,0x1.0267760000000p-24f,0x1.0e1bd80000000p-24f,0x1.1a41600000000p-24f,0x1.26dbc80000000p-24f,0x1.33ee7a0000000p-24f,0x1.417ca40000000p-24f,0x1.4f8a220000000p-24f,
0x1.5e1a780000000p-24f,0x1.6d317e0000000p-24f,0x1.7cd2f20000000p-24f,0x1.8d02ae0000000p-24f,0x1.9dc4c00000000p-24f,0x1.af1c940000000p-24f,0x1.c10f700000000p-24f,0x1.d3a0a00000000p-24f,
0x1.e6d48c0000000p-24f,0x1.faaf960000000p-24f,0x1.079b140000000p-23f,0x1.1236520000000p-23f,0x1.1d2bba0000000p-23f,0x1.287d7c0000000p-23f,0x1.342e5c0000000p-23f,0x1.4040880000000p-23f,
0x1.4cb6380000000p-23f,0x1.5992b40000000p-23f,0x1.66d7a60000000p-23f,0x1.7487ca0000000p-23f,0x1.82a5560000000p-23f,0x1.9134220000000p-23f,0x1.a035460000000p-23f,0x1.afac9e0000000p-23f,
0x1.bf9ce80000000p-23f,0x1.d008560000000p-23f,0x1.e0f1aa0000000p-23f,0x1.f25c2e0000000p-23f,0x1.0225980000000p-22f,0x1.0b602c0000000p-22f,0x1.14e0040000000p-22f,0x1.1ea63c0000000p-22f,
0x1.28b4760000000p-22f,0x1.330c5c0000000p-22f,0x1.3daf940000000p-22f,0x1.489f360000000p-22f,0x1.53dd740000000p-22f,0x1.5f6bb00000000p-22f,0x1.6b4b900000000p-22f,0x1.777eba0000000p-22f,
0x1.84071a0000000p-22f,0x1.90e6560000000p-22f,0x1.9e1dd00000000p-22f,0x1.abaffe0000000p-22f,0x1.b99db60000000p-22f,0x1.c7e9b60000000p-22f,0x1.d6955e0000000p-22f,0x1.e5a29a0000000p-22f,
0x1.f513120000000p-22f,0x1.02747c0000000p-21f,0x1.0a93400000000p-21f,0x1.42724c0000000p-42f,0x1.7779f20000000p-42f,0x1.b471a00000000p-42f,0x1.fa65600000000p-42f,0x1.2541ec0000000p-41f,
0x1.5312040000000p-41f,0x1.875fc20000000p-41f,0x1.c2fc1e0000000p-41f,0x1.036a700000000p-40f,0x1.29f82c0000000p-40f,0x1.55aeb60000000p-40f,0x1.8738da0000000p-40f,0x1.bf43840000000p-40f,
0x1.fe8a100000000p-40f,0x1.22f5f00000000p-39f,0x1.4b2b5c0000000p-39f,0x1.78636a0000000p-39f,0x1.ab2a9a0000000p-39f,0x1.e426aa0000000p-39f,0x1.11fcdc0000000p-38f,0x1.35b32c0000000p-38f,
0x1.5d99560000000p-38f,0x1.8a22a60000000p-38f,0x1.bbc7ce0000000p-38f,0x1.f30c500000000p-38f,0x1.1841f20000000p-37f,0x1.3a65a20000000p-37f,0x1.6044140000000p-37f,0x1.8a3daa0000000p-37f,
0x1.b8b82e0000000p-37f,0x1.ec16b00000000p-37f,0x1.1270280000000p-36f,0x1.31c0900000000p-36f,0x1.5444a00000000p-36f,0x1.7a48e80000000p-36f,0x1.a41e760000000p-36f,0x1.d2165e0000000p-36f,
0x1.0245580000000p-35f,0x1.1df3c20000000p-35f,0x1.3c4a380000000p-35f,0x1.5d7ec40000000p-35f,0x1.81d2b20000000p-35f,0x1.a985120000000p-35f,0x1.d4dbb00000000p-35f,0x1.0212ae0000000p-34f,
0x1.1bd8740000000p-34f,0x1.37e8d20000000p-34f,0x1.5673120000000p-34f,0x1.77a67e0000000p-34f,0x1.9bb91e0000000p-34f,0x1.c2e3420000000p-34f,0x1.ed54320000000p-34f,0x1.0db02e0000000p-33f,
0x1.269a680000000p-33f,0x1.4195ce0000000p-33f,0x1.5ebe840000000p-33f,0x1.7e41980000000p-33f,0x1.a0466e0000000p-33f,0x1.c4fa100000000p-33f,0x1.ec8f280000000p-33f,0x1.0b9c320000000p-32f,
0x1.2291640000000p-32f,0x1.3b46220000000p-32f,0x1.55d6920000000p-32f,0x1.7261aa0000000p-32f,0x1.91092e0000000p-32f,0x1.b1eeea0000000p-32f,0x1.d53a420000000p-32f,0x1.fb0a300000000p-32f,
0x1.11c7440000000p-31f,0x1.2775f40000000p-31f,0x1.3ea9140000000p-31f,0x1.5777280000000p-31f,0x1.71f9840000000p-31f,0x1.8e497e0000000p-31f,0x1.ac81d60000000p-31f,0x1.ccbeae0000000p-31f,
0x1.ef1f000000000p-31f,0x1.09e02a0000000p-30f,0x1.1d60d40000000p-30f,0x1.3228b60000000p-30f,0x1.483b520000000p-30f,0x1.5fbbd80000000p-30f,0x1.78b1540000000p-30f,0x1.9337ea0000000p-30f,
0x1.af64b60000000p-30f,0x1.cd45ca0000000p-30f,0x1.ecf74e0000000p-30f,0x1.074ab40000000p-29f,0x1.191a980000000p-29f,0x1.2bf5e40000000p-29f,0x1.3fee2e0000000p-29f,0x1.550e040000000p-29f,
0x1.6b6a840000000p-29f,0x1.830ab60000000p-29f,0x1.9c03b60000000p-29f,0x1.b6671c0000000p-29f,0x1.d242fc0000000p-29f,0x1.efa8ec0000000p-29f,0x1.0757040000000p-28f,0x1.17b0320000000p-28f,
0x1.28ee140000000p-28f,0x1.3b17b20000000p-28f,0x1.4e395c0000000p-28f,0x1.625da00000000p-28f,0x1.7790d00000000p-28f,0x1.8ddd7a0000000p-28f,0x1.a551b20000000p-28f,0x1.bdf8020000000p-28f,
0x1.d7e0440000000p-28f,0x1.f315020000000p-28f,0x1.07d3ec0000000p-27f,0x1.16d28a0000000p-27f,0x1.268e440000000p-27f,0x1.370f080000000p-27f,0x1.485cbe0000000p-27f,0x1.5a7bce0000000p-27f,
0x1.6d84d60000000p-27f,0x1.8169c60000000p-27f,0x1.9645000000000p-27f,0x1.ac0dba0000000p-27f,0x1.c2e7200000000p-27f,0x1.dac86a0000000p-27f,0x1.f3c32c0000000p-27f,0x1.06f01a0000000p-26f,
0x1.1494260000000p-26f,0x1.22d2220000000p-26f,0x1.31b73c0000000p-26f,0x1.413f120000000p-26f,0x1.5176d40000000p-26f,0x1.62674e0000000p-26f,0x1.740c180000000p-26f,0x1.8676ce0000000p-26f,
0x1.99a76c0000000p-26f,0x1.adab240000000p-26f,0x1.c281fa0000000p-26f,0x1.d8391a0000000p-26f,0x1.eed9540000000p-26f,0x1.0331540000000p-25f,0x1.0f71220000000p-25f,0x1.1c307c0000000p-25f,
0x1.2973c60000000p-25f,0x1.373d360000000p-25f,0x1.4593620000000p-25f,0x1.547ab00000000p-25f,0x1.63f7860000000p-25f,0x1.740e4c0000000p-25f,0x1.84c5980000000p-25f,0x1.961f9e0000000p-25f,
0x1.a822f80000000p-25f,0x1.bad40c0000000p-25f,0x1.ce39700000000p-25f,0x1.e259c00000000p-25f,0x1.f739620000000p-25f,0x1.066e5c0000000p-24f,0x1.11a5300000000p-24f,0x1.1d45900000000p-24f,
0x1.2950980000000p-24f,0x1.35caac0000000p-24f,0x1.42b71a0000000p-24f,0x1.50192e0000000p-24f,0x1.5df31c0000000p-24f,0x1.6c4a600000000p-24f,0x1.7b20180000000p-24f,0x1.8a79be0000000p-24f,
0x1.9a5aa40000000p-24f,0x1.aac6120000000p-24f,0x1.bbc18a0000000p-24f,0x1.cd478c0000000p-24f,0x1.df727c0000000p-24f,0x1.f22c5a0000000p-24f,0x1.02c5940000000p-23f,0x1.0cc1f00000000p-23f,
0x1.1716440000000p-23f,0x1.21c28e0000000p-23f,0x1.2cc14e0000000p-23f,0x1.381d820000000p-23f,0x1.43d72e0000000p-23f,0x1.4fee4e0000000p-23f,0x1.5c68620000000p-23f,0x1.69456c0000000p-23f
};
extern "C" __device__ const float mpas_static_totplk16[181] = {
0x1.40aae40000000p-42f,0x1.7556500000000p-42f,0x1.b1dd940000000p-42f,0x1.f749dc0000000p-42f,0x1.2366580000000p-41f,0x1.50d8c40000000p-41f,0x1.84b6180000000p-41f,0x1.bfd2300000000p-41f,
0x1.0187a80000000p-40f,0x1.27b7b60000000p-40f,0x1.5309600000000p-40f,0x1.8419020000000p-40f,0x1.bb8c5c0000000p-40f,0x1.fa34600000000p-40f,0x1.2069d00000000p-39f,0x1.4828580000000p-39f,
0x1.74deb20000000p-39f,0x1.a70e900000000p-39f,0x1.df5a160000000p-39f,0x1.0f31b00000000p-38f,0x1.32711a0000000p-38f,0x1.59d1f60000000p-38f,0x1.85c05a0000000p-38f,0x1.b6b4f80000000p-38f,
0x1.ed2fb60000000p-38f,0x1.14df740000000p-37f,0x1.367ebc0000000p-37f,0x1.5bc8900000000p-37f,0x1.851a9e0000000p-37f,0x1.b2cef80000000p-37f,0x1.e5554e0000000p-37f,0x1.0e8cdc0000000p-36f,
0x1.2d51a80000000p-36f,0x1.4f38180000000p-36f,0x1.74883a0000000p-36f,0x1.9d8e9e0000000p-36f,0x1.caa0d80000000p-36f,0x1.fc14760000000p-36f,0x1.1926480000000p-35f,0x1.36d85a0000000p-35f,
0x1.5758c20000000p-35f,0x1.7adfc60000000p-35f,0x1.a1ac760000000p-35f,0x1.cc06e00000000p-35f,0x1.fa32940000000p-35f,0x1.1640500000000p-34f,0x1.319fea0000000p-34f,0x1.4f66400000000p-34f,
0x1.6fbf400000000p-34f,0x1.92dfd00000000p-34f,0x1.b8fdfc0000000p-34f,0x1.e24a300000000p-34f,0x1.07824a0000000p-33f,0x1.1fb8600000000p-33f,0x1.39e91c0000000p-33f,0x1.5636460000000p-33f,
0x1.74c1a60000000p-33f,0x1.95bde60000000p-33f,0x1.b94ccc0000000p-33f,0x1.df9b620000000p-33f,0x1.046e2a0000000p-32f,0x1.1a9e540000000p-32f,0x1.327d260000000p-32f,0x1.4c21260000000p-32f,
0x1.67ac180000000p-32f,0x1.853a240000000p-32f,0x1.a4ea400000000p-32f,0x1.c6e3d40000000p-32f,0x1.eb45d60000000p-32f,0x1.091bd60000000p-31f,0x1.1df1ca0000000p-31f,0x1.3437100000000p-31f,
0x1.4c02300000000p-31f,0x1.656c7a0000000p-31f,0x1.808b0e0000000p-31f,0x1.9d77420000000p-31f,0x1.bc4d3a0000000p-31f,0x1.dd27b20000000p-31f,0x1.00116a0000000p-30f,0x1.12aec80000000p-30f,
0x1.267a0a0000000p-30f,0x1.3b88480000000p-30f,0x1.51e84c0000000p-30f,0x1.69a8280000000p-30f,0x1.82dcf80000000p-30f,0x1.9d9bda0000000p-30f,0x1.b9f9e80000000p-30f,0x1.d805360000000p-30f,
0x1.f7d9e80000000p-30f,0x1.0cca140000000p-29f,0x1.1ea1020000000p-29f,0x1.317fd20000000p-29f,0x1.4574980000000p-29f,0x1.5a89e00000000p-29f,0x1.70d1440000000p-29f,0x1.88554e0000000p-29f,
0x1.a1279c0000000p-29f,0x1.bb563a0000000p-29f,0x1.d6f2c40000000p-29f,0x1.f40ed20000000p-29f,0x1.095c3a0000000p-28f,0x1.1982640000000p-28f,0x1.2a80f00000000p-28f,0x1.3c64300000000p-28f,
0x1.4f34ec0000000p-28f,0x1.62fdb60000000p-28f,0x1.77c75a0000000p-28f,0x1.8d9fe80000000p-28f,0x1.a4902e0000000p-28f,0x1.bca63e0000000p-28f,0x1.d5eae20000000p-28f,0x1.f06df40000000p-28f,
0x1.061cfe0000000p-27f,0x1.14af6c0000000p-27f,0x1.23f3860000000p-27f,0x1.33f21a0000000p-27f,0x1.44b1520000000p-27f,0x1.5638340000000p-27f,0x1.6892340000000p-27f,0x1.7bbe6c0000000p-27f,
0x1.8fcf560000000p-27f,0x1.a4c4f40000000p-27f,0x1.bab0dc0000000p-27f,0x1.d19bdc0000000p-27f,0x1.e97d240000000p-27f,0x1.013bf40000000p-26f,0x1.0e41ac0000000p-26f,0x1.1bd8860000000p-26f,
0x1.2a04ea0000000p-26f,0x1.38cb3c0000000p-26f,0x1.482fe20000000p-26f,0x1.583ba80000000p-26f,0x1.68f75a0000000p-26f,0x1.7a5e940000000p-26f,0x1.8c7e840000000p-26f,0x1.9f5b920000000p-26f,
0x1.b2fe8a0000000p-26f,0x1.c7676a0000000p-26f,0x1.dca3680000000p-26f,0x1.f2b6e40000000p-26f,0x1.04d5580000000p-25f,0x1.10c1960000000p-25f,0x1.1d22600000000p-25f,0x1.29fe500000000p-25f,
0x1.3759cc0000000p-25f,0x1.4534d20000000p-25f,0x1.5395fc0000000p-25f,0x1.6281b00000000p-25f,0x1.71fa220000000p-25f,0x1.8205e80000000p-25f,0x1.92a7380000000p-25f,0x1.a3e4a80000000p-25f,
0x1.b5c06c0000000p-25f,0x1.c8411e0000000p-25f,0x1.db68f20000000p-25f,0x1.ef40b00000000p-25f,0x1.01e5460000000p-24f,0x1.0c85780000000p-24f,0x1.1784380000000p-24f,0x1.22e3ba0000000p-24f,
0x1.2ea6300000000p-24f,0x1.3ad0000000000p-24f,0x1.4762460000000p-24f,0x1.5461660000000p-24f,0x1.61ce780000000p-24f,0x1.6facca0000000p-24f,0x1.7e00c40000000p-24f,0x1.8ccb7c0000000p-24f,
0x1.9c11580000000p-24f,0x1.abd48e0000000p-24f,0x1.bc19800000000p-24f,0x1.cce4980000000p-24f,0x1.de33a00000000p-24f,0x1.f011980000000p-24f,0x1.013f400000000p-23f,0x1.0abd2c0000000p-23f,
0x1.1482900000000p-23f,0x1.1e9a6a0000000p-23f,0x1.28ff3a0000000p-23f,0x1.33b1040000000p-23f,0x1.3eb5420000000p-23f
};
extern "C" __device__ const float mpas_static_preflog[59] = {
0x1.bd70a40000000p+2f,0x1.b0a3d80000000p+2f,0x1.a3d70a0000000p+2f,0x1.970a3e0000000p+2f,0x1.8a3d700000000p+2f,0x1.7d70a40000000p+2f,0x1.70a3d80000000p+2f,0x1.63d70a0000000p+2f,
0x1.570a3e0000000p+2f,0x1.4a3d700000000p+2f,0x1.3d70a40000000p+2f,0x1.30a3d80000000p+2f,0x1.23d70a0000000p+2f,0x1.170a3e0000000p+2f,0x1.0a3d700000000p+2f,0x1.fae1480000000p+1f,
0x1.e147ae0000000p+1f,0x1.c7ae140000000p+1f,0x1.ae147a0000000p+1f,0x1.947ae20000000p+1f,0x1.7ae1480000000p+1f,0x1.6147ae0000000p+1f,0x1.47ae140000000p+1f,0x1.2e147a0000000p+1f,
0x1.147ae20000000p+1f,0x1.f5c2900000000p+0f,0x1.c28f5c0000000p+0f,0x1.8f5c280000000p+0f,0x1.5c28f60000000p+0f,0x1.28f5c20000000p+0f,0x1.eb851e0000000p-1f,0x1.851eb80000000p-1f,
0x1.1eb8520000000p-1f,0x1.70a3d80000000p-2f,0x1.47ae140000000p-3f,-0x1.47ae140000000p-5f,-0x1.eb851e0000000p-3f,-0x1.c28f5c0000000p-2f,-0x1.47ae140000000p-1f,-0x1.ae147a0000000p-1f,
-0x1.0a3d700000000p+0f,-0x1.3d70a40000000p+0f,-0x1.70a3d80000000p+0f,-0x1.a3d70a0000000p+0f,-0x1.d70a3e0000000p+0f,-0x1.051eb80000000p+1f,-0x1.1eb8520000000p+1f,-0x1.3851ec0000000p+1f,
-0x1.51eb860000000p+1f,-0x1.6b851e0000000p+1f,-0x1.851eb80000000p+1f,-0x1.9eb8520000000p+1f,-0x1.b851ec0000000p+1f,-0x1.d1eb860000000p+1f,-0x1.eb851e0000000p+1f,-0x1.028f5c0000000p+2f,
-0x1.0f5c280000000p+2f,-0x1.1c28f60000000p+2f,-0x1.28f5c20000000p+2f
};
extern "C" __device__ const float mpas_static_tref[59] = {
0x1.2633340000000p+8f,0x1.1ffd700000000p+8f,0x1.16f0a40000000p+8f,0x1.0d40000000000p+8f,0x1.03d47a0000000p+8f,0x1.f4570a0000000p+7f,0x1.e18a3e0000000p+7f,0x1.cf947a0000000p+7f,
0x1.be1eb80000000p+7f,0x1.af8f5c0000000p+7f,0x1.af66660000000p+7f,0x1.af66660000000p+7f,0x1.af66660000000p+7f,0x1.b21eb80000000p+7f,0x1.b528f60000000p+7f,0x1.b85c280000000p+7f,
0x1.bb7ae20000000p+7f,0x1.be8f5c0000000p+7f,0x1.c1947a0000000p+7f,0x1.c5199a0000000p+7f,0x1.c8ae140000000p+7f,0x1.ce42900000000p+7f,0x1.d4051e0000000p+7f,0x1.da0f5c0000000p+7f,
0x1.e070a40000000p+7f,0x1.e76b860000000p+7f,0x1.ee851e0000000p+7f,0x1.f5b3340000000p+7f,0x1.fd23d80000000p+7f,0x1.0251ec0000000p+8f,0x1.0628f60000000p+8f,0x1.0a0f5c0000000p+8f,
0x1.0dfd700000000p+8f,0x1.1166660000000p+8f,0x1.135c280000000p+8f,0x1.13ae140000000p+8f,0x1.11b8520000000p+8f,0x1.0fa1480000000p+8f,0x1.0d8ccc0000000p+8f,0x1.09ee140000000p+8f,
0x1.061c280000000p+8f,0x1.0247ae0000000p+8f,0x1.fb33340000000p+7f,0x1.f1147a0000000p+7f,0x1.e6f5c20000000p+7f,0x1.dc2e140000000p+7f,0x1.d01eb80000000p+7f,0x1.c40f5c0000000p+7f,
0x1.b800000000000p+7f,0x1.acb3340000000p+7f,0x1.a1bd700000000p+7f,0x1.96cccc0000000p+7f,0x1.8bd70a0000000p+7f,0x1.81cccc0000000p+7f,0x1.782e140000000p+7f,0x1.6e947a0000000p+7f,
0x1.64fae20000000p+7f,0x1.5be1480000000p+7f,0x1.583d700000000p+7f
};
extern "C" __device__ const float mpas_static_absice1[10] = {
0x1.d7dbf40000000p-9f,0x1.22d0e60000000p+0f,0x1.bda5120000000p-8f,0x1.3333340000000p-1f,0x1.3a92a40000000p-12f,0x1.56872c0000000p+0f,0x1.a36e2e0000000p-10f,0x1.2a7efa0000000p+0f,
0x1.0624de0000000p-9f,0x1.1e35400000000p+0f
};
extern "C" __device__ const float mpas_static_absice2[688] = {
0x1.3f726e0000000p-4f,0x1.03b4be0000000p-4f,0x1.bbd7220000000p-5f,0x1.8673660000000p-5f,0x1.5e043c0000000p-5f,0x1.3ded300000000p-5f,0x1.23990c0000000p-5f,0x1.0d74620000000p-5f,
0x1.f4f14e0000000p-6f,0x1.d3e4f00000000p-6f,0x1.b6c4620000000p-6f,0x1.9cd4620000000p-6f,0x1.8587ca0000000p-6f,0x1.7071c20000000p-6f,0x1.5d3cdc0000000p-6f,0x1.4ba4da0000000p-6f,
0x1.3b727c0000000p-6f,0x1.2c786e0000000p-6f,0x1.1e91300000000p-6f,0x1.119d2c0000000p-6f,0x1.0581d40000000p-6f,0x1.f4510a0000000p-7f,0x1.defb8e0000000p-7f,0x1.cae1900000000p-7f,
0x1.b7e5e00000000p-7f,0x1.a5eefc0000000p-7f,0x1.94e6560000000p-7f,0x1.84b84c0000000p-7f,0x1.7553280000000p-7f,0x1.66a7540000000p-7f,0x1.58a6e00000000p-7f,0x1.4b45220000000p-7f,
0x1.3e76d40000000p-7f,0x1.3231ce0000000p-7f,0x1.266cd00000000p-7f,0x1.1b1f720000000p-7f,0x1.10420e0000000p-7f,0x1.05cdac0000000p-7f,0x1.f777ce0000000p-8f,0x1.e40dca0000000p-8f,
0x1.d152840000000p-8f,0x1.bf3c0c0000000p-8f,0x1.adc1440000000p-8f,0x1.c8464a0000000p-6f,0x1.bbfbe40000000p-6f,0x1.ad1fbe0000000p-6f,0x1.9e63980000000p-6f,0x1.904c480000000p-6f,
0x1.82eba20000000p-6f,0x1.7636c80000000p-6f,0x1.6a1c680000000p-6f,0x1.5e8b260000000p-6f,0x1.5373440000000p-6f,0x1.48c72e0000000p-6f,0x1.3e7b3a0000000p-6f,0x1.34856e0000000p-6f,
0x1.2add240000000p-6f,0x1.217b040000000p-6f,0x1.1858b20000000p-6f,0x1.0f70960000000p-6f,0x1.06bdd40000000p-6f,0x1.fc784c0000000p-7f,0x1.ebcf720000000p-7f,0x1.db7a7a0000000p-7f,
0x1.cb735c0000000p-7f,0x1.bbb4cc0000000p-7f,0x1.ac39da0000000p-7f,0x1.9cfe3e0000000p-7f,0x1.8dfdf00000000p-7f,0x1.7f356a0000000p-7f,0x1.70a1680000000p-7f,0x1.623ee40000000p-7f,
0x1.540b300000000p-7f,0x1.4603aa0000000p-7f,0x1.38260e0000000p-7f,0x1.2a70360000000p-7f,0x1.1ce0280000000p-7f,0x1.0f74080000000p-7f,0x1.022a220000000p-7f,0x1.ea01bc0000000p-8f,
0x1.cfed7a0000000p-8f,0x1.b614be0000000p-8f,0x1.9c74e60000000p-8f,0x1.830b8a0000000p-8f,0x1.69d65e0000000p-8f,0x1.50d33a0000000p-8f,0x1.b462f80000000p-4f,0x1.47ea200000000p-4f,
0x1.0c24460000000p-4f,0x1.c9dc920000000p-5f,0x1.914cca0000000p-5f,0x1.65fc1e0000000p-5f,0x1.43697c0000000p-5f,0x1.26f7a60000000p-5f,0x1.0f04100000000p-5f,0x1.f4f3680000000p-6f,
0x1.d131ec0000000p-6f,0x1.b1b0e80000000p-6f,0x1.95a59a0000000p-6f,0x1.7c769e0000000p-6f,0x1.65ad7a0000000p-6f,0x1.50ecec0000000p-6f,0x1.3dea920000000p-6f,0x1.2c6a2c0000000p-6f,
0x1.1c3a3e0000000p-6f,0x1.0d31d80000000p-6f,0x1.fe5d940000000p-7f,0x1.e428340000000p-7f,0x1.cb92500000000p-7f,0x1.b471e20000000p-7f,0x1.9ea26c0000000p-7f,0x1.8a04660000000p-7f,
0x1.767c340000000p-7f,0x1.63f1d80000000p-7f,0x1.524fc40000000p-7f,0x1.4183440000000p-7f,0x1.317b880000000p-7f,0x1.2229920000000p-7f,0x1.137ffc0000000p-7f,0x1.0572b80000000p-7f,
0x1.efedc60000000p-8f,0x1.d605560000000p-8f,0x1.bd1a4a0000000p-8f,0x1.a51c620000000p-8f,0x1.8dfcd40000000p-8f,0x1.77ae1e0000000p-8f,0x1.6223d80000000p-8f,0x1.4d52a60000000p-8f,
0x1.39300c0000000p-8f,0x1.71933a0000000p-3f,0x1.ded12c0000000p-4f,0x1.638d080000000p-4f,0x1.1b093c0000000p-4f,0x1.d6123c0000000p-5f,0x1.91995a0000000p-5f,0x1.5e2c8c0000000p-5f,
0x1.360f960000000p-5f,0x1.15d9d20000000p-5f,0x1.f6c6300000000p-6f,0x1.ca79400000000p-6f,0x1.a4d1dc0000000p-6f,0x1.84662c0000000p-6f,0x1.682cf00000000p-6f,0x1.4f5ee20000000p-6f,
0x1.3963ae0000000p-6f,0x1.25c48a0000000p-6f,0x1.1423a40000000p-6f,0x1.0435a40000000p-6f,0x1.eb7b380000000p-7f,0x1.d112ee0000000p-7f,0x1.b8dfb00000000p-7f,0x1.a29c220000000p-7f,
0x1.8e0e360000000p-7f,0x1.7b046e0000000p-7f,0x1.6954ae0000000p-7f,0x1.58dab80000000p-7f,0x1.4977580000000p-7f,0x1.3b0f3e0000000p-7f,0x1.2d8ac80000000p-7f,0x1.20d5220000000p-7f,
0x1.14dbfe0000000p-7f,0x1.098f260000000p-7f,0x1.fdc0760000000p-8f,0x1.e984d80000000p-8f,0x1.d654900000000p-8f,0x1.c41b1a0000000p-8f,0x1.b2c6040000000p-8f,0x1.a244a00000000p-8f,
0x1.9287e80000000p-8f,0x1.8382380000000p-8f,0x1.7527280000000p-8f,0x1.676b6c0000000p-8f,0x1.b498080000000p-3f,0x1.0c91aa0000000p-3f,0x1.8151560000000p-4f,0x1.2afc320000000p-4f,
0x1.e6b5860000000p-5f,0x1.991cac0000000p-5f,0x1.5ff2ba0000000p-5f,0x1.341c420000000p-5f,0x1.11710c0000000p-5f,0x1.eab27c0000000p-6f,0x1.bc42e60000000p-6f,0x1.95407a0000000p-6f,
0x1.7407440000000p-6f,0x1.5766380000000p-6f,0x1.3e7a840000000p-6f,0x1.28981a0000000p-6f,0x1.153a040000000p-6f,0x1.03f7d20000000p-6f,0x1.e8fd1a0000000p-7f,0x1.cd16ca0000000p-7f,
0x1.b3cf900000000p-7f,0x1.9cce020000000p-7f,0x1.87c7b60000000p-7f,0x1.747e6c0000000p-7f,0x1.62bda40000000p-7f,0x1.5258be0000000p-7f,0x1.4329c80000000p-7f,0x1.350fe60000000p-7f,
0x1.27eed60000000p-7f,0x1.1bae000000000p-7f,0x1.1037ee0000000p-7f,0x1.0579d00000000p-7f,0x1.f6c62c0000000p-8f,0x1.e3ca360000000p-8f,0x1.d1e5c40000000p-8f,0x1.c101b40000000p-8f,
0x1.b109620000000p-8f,0x1.a1ea500000000p-8f,0x1.9393dc0000000p-8f,0x1.85f70c0000000p-8f,0x1.79065a0000000p-8f,0x1.6cb5860000000p-8f,0x1.60f9760000000p-8f,0x1.3ca8340000000p-3f,
0x1.a9ad5e0000000p-4f,0x1.4402000000000p-4f,0x1.06a7320000000p-4f,0x1.ba5e260000000p-5f,0x1.7e26d20000000p-5f,0x1.5038160000000p-5f,0x1.2be4240000000p-5f,0x1.0e5a960000000p-5f,
0x1.eb9db20000000p-6f,0x1.c217be0000000p-6f,0x1.9e74800000000p-6f,0x1.7f7dda0000000p-6f,0x1.644e7e0000000p-6f,0x1.4c39340000000p-6f,0x1.36b86e0000000p-6f,0x1.23635c0000000p-6f,
0x1.11e6be0000000p-6f,0x1.01ff5e0000000p-6f,0x1.e6ecde0000000p-7f,0x1.cc3d9c0000000p-7f,0x1.b3a59e0000000p-7f,0x1.9ce6000000000p-7f,0x1.87c9380000000p-7f,0x1.7421d00000000p-7f,
0x1.61c8c80000000p-7f,0x1.509c900000000p-7f,0x1.407fe40000000p-7f,0x1.31594a0000000p-7f,0x1.2312740000000p-7f,0x1.1597be0000000p-7f,0x1.08d7c80000000p-7f,0x1.f986580000000p-8f,
0x1.e2985a0000000p-8f,0x1.cccd080000000p-8f,0x1.b80e640000000p-8f,0x1.a448a40000000p-8f,0x1.9169e40000000p-8f,0x1.7f62020000000p-8f,0x1.6e22580000000p-8f,0x1.5d9d880000000p-8f,
0x1.4dc7780000000p-8f,0x1.3e950c0000000p-8f,0x1.369dc20000000p-4f,0x1.fa64aa0000000p-5f,0x1.b329480000000p-5f,0x1.80b3140000000p-5f,0x1.5a315a0000000p-5f,0x1.3b5ecc0000000p-5f,
0x1.21dc5e0000000p-5f,0x1.0c38920000000p-5f,0x1.f306080000000p-6f,0x1.d22e560000000p-6f,0x1.b4ffc60000000p-6f,0x1.9ace020000000p-6f,0x1.8316e00000000p-6f,0x1.6d75f00000000p-6f,
0x1.599c1e0000000p-6f,0x1.474a220000000p-6f,0x1.364c980000000p-6f,0x1.2679660000000p-6f,0x1.17ad700000000p-6f,0x1.09cb500000000p-6f,0x1.f974360000000p-7f,0x1.e0c9180000000p-7f,
0x1.c970b80000000p-7f,0x1.b34b720000000p-7f,0x1.9e3dc40000000p-7f,0x1.8a2f6e0000000p-7f,0x1.770b500000000p-7f,0x1.64be860000000p-7f,0x1.5338620000000p-7f,0x1.426a020000000p-7f,
0x1.3246140000000p-7f,0x1.22c0980000000p-7f,0x1.13cebe0000000p-7f,0x1.0566bc0000000p-7f,0x1.eeff6c0000000p-8f,0x1.d4232e0000000p-8f,0x1.ba2a000000000p-8f,0x1.a106680000000p-8f,
0x1.88ac0c0000000p-8f,0x1.710f960000000p-8f,0x1.5a26920000000p-8f,0x1.43e7580000000p-8f,0x1.2e48fe0000000p-8f,0x1.718c780000000p-4f,0x1.1b8dc60000000p-4f,0x1.d3d1840000000p-5f,
0x1.91466a0000000p-5f,0x1.60be3c0000000p-5f,0x1.3b5afa0000000p-5f,0x1.1d6c9e0000000p-5f,0x1.04c4f60000000p-5f,0x1.e004780000000p-6f,0x1.bc6fd20000000p-6f,0x1.9d82320000000p-6f,
0x1.824db80000000p-6f,0x1.6a221e0000000p-6f,0x1.54795e0000000p-6f,0x1.40ebce0000000p-6f,0x1.2f27a00000000p-6f,0x1.1eeb680000000p-6f,0x1.10020c0000000p-6f,0x1.023ff60000000p-6f,
0x1.eb02420000000p-7f,0x1.d34e860000000p-7f,0x1.bd31a40000000p-7f,0x1.a8803c0000000p-7f,0x1.9514fa0000000p-7f,0x1.82cfd40000000p-7f,0x1.7194ee0000000p-7f,0x1.614be00000000p-7f,
0x1.51df700000000p-7f,0x1.433cdc0000000p-7f,0x1.3553700000000p-7f,0x1.28146a0000000p-7f,0x1.1b72a20000000p-7f,0x1.0f624a0000000p-7f,0x1.03d8c80000000p-7f,0x1.f199240000000p-8f,
0x1.dc6a0e0000000p-8f,0x1.c814a40000000p-8f,0x1.b48ab60000000p-8f,0x1.a1bf4c0000000p-8f,0x1.8fa6920000000p-8f,0x1.7e35b60000000p-8f,0x1.6d62ce0000000p-8f,0x1.5d24b00000000p-8f,
0x1.09076e0000000p-3f,0x1.67f7220000000p-4f,0x1.1397360000000p-4f,0x1.c0e60e0000000p-5f,0x1.7bb4060000000p-5f,0x1.4980c20000000p-5f,0x1.2345d60000000p-5f,0x1.0519aa0000000p-5f,
0x1.d940640000000p-6f,0x1.b0acfc0000000p-6f,0x1.8e71720000000p-6f,0x1.7123440000000p-6f,0x1.57bd820000000p-6f,0x1.417f720000000p-6f,0x1.2dd75e0000000p-6f,0x1.1c550a0000000p-6f,
0x1.0ca0980000000p-6f,0x1.fce81e0000000p-7f,0x1.e32e640000000p-7f,0x1.cbb83a0000000p-7f,0x1.b63a240000000p-7f,0x1.a2754a0000000p-7f,0x1.9035160000000p-7f,0x1.7f4cfc0000000p-7f,
0x1.6f97280000000p-7f,0x1.60f3200000000p-7f,0x1.53450c0000000p-7f,0x1.4674b40000000p-7f,0x1.3a6cf60000000p-7f,0x1.2f1b5c0000000p-7f,0x1.246fb00000000p-7f,0x1.1a5b980000000p-7f,
0x1.10d2640000000p-7f,0x1.07c8c60000000p-7f,0x1.fe694a0000000p-8f,0x1.ee19f00000000p-8f,0x1.de93400000000p-8f,0x1.cfc6960000000p-8f,0x1.c1a6ba0000000p-8f,0x1.b427ae0000000p-8f,
0x1.a73e900000000p-8f,0x1.9ae17a0000000p-8f,0x1.8f07620000000p-8f,0x1.4660000000000p-3f,0x1.9f8f7c0000000p-4f,0x1.3184d00000000p-4f,0x1.e39edc0000000p-5f,0x1.9064060000000p-5f,
0x1.55b38c0000000p-5f,0x1.2a10740000000p-5f,0x1.0854a20000000p-5f,0x1.daed000000000p-6f,0x1.af1ae00000000p-6f,0x1.8aada40000000p-6f,0x1.6bea080000000p-6f,0x1.5195740000000p-6f,
0x1.3aca360000000p-6f,0x1.26dcd80000000p-6f,0x1.154ac00000000p-6f,0x1.05ae760000000p-6f,0x1.ef707c0000000p-7f,0x1.d650c40000000p-7f,0x1.bf96c40000000p-7f,0x1.aaee560000000p-7f,
0x1.9812300000000p-7f,0x1.86c8860000000p-7f,0x1.76e0c80000000p-7f,0x1.6832180000000p-7f,0x1.5a99940000000p-7f,0x1.4df95e0000000p-7f,0x1.4237b80000000p-7f,0x1.373e500000000p-7f,
0x1.2cf9c20000000p-7f,0x1.23591a0000000p-7f,0x1.1a4d680000000p-7f,0x1.11c98c0000000p-7f,0x1.09c1da0000000p-7f,0x1.022bfa0000000p-7f,0x1.f5fd6e0000000p-8f,0x1.e863920000000p-8f,
0x1.db7b960000000p-8f,0x1.cf38260000000p-8f,0x1.c38d340000000p-8f,0x1.b86fe20000000p-8f,0x1.add64e0000000p-8f,0x1.a3b7840000000p-8f,0x1.5331fe0000000p-3f,0x1.a6c2920000000p-4f,
0x1.32aea60000000p-4f,0x1.e0fcd60000000p-5f,0x1.8b89f20000000p-5f,0x1.4fdf000000000p-5f,0x1.23dfee0000000p-5f,0x1.021c200000000p-5f,0x1.cec8040000000p-6f,0x1.a374ba0000000p-6f,
0x1.7fa1c20000000p-6f,0x1.61862c0000000p-6f,0x1.47df1c0000000p-6f,0x1.31c2100000000p-6f,0x1.1e807e0000000p-6f,0x1.0d95f60000000p-6f,0x1.fd383a0000000p-7f,0x1.e285300000000p-7f,
0x1.ca92f60000000p-7f,0x1.b4fa880000000p-7f,0x1.a167da0000000p-7f,0x1.8f960a0000000p-7f,0x1.7f4ba60000000p-7f,0x1.7058ae0000000p-7f,0x1.6294ca0000000p-7f,0x1.55dd820000000p-7f,
0x1.4a15780000000p-7f,0x1.3f237e0000000p-7f,0x1.34f1aa0000000p-7f,0x1.2b6d0e0000000p-7f,0x1.2285180000000p-7f,0x1.1a2b440000000p-7f,0x1.1252ca0000000p-7f,0x1.0af05e0000000p-7f,
0x1.03f9fa0000000p-7f,0x1.facd5a0000000p-8f,0x1.ee5d040000000p-8f,0x1.e294a80000000p-8f,0x1.d767740000000p-8f,0x1.ccc9d00000000p-8f,0x1.c2b1520000000p-8f,0x1.b914800000000p-8f,
0x1.afeac40000000p-8f,0x1.789c460000000p-4f,0x1.08104c0000000p-4f,0x1.967d660000000p-5f,0x1.4b202a0000000p-5f,0x1.180a380000000p-5f,0x1.e66ad80000000p-6f,0x1.aee2000000000p-6f,
0x1.8393480000000p-6f,0x1.60e5be0000000p-6f,0x1.4488660000000p-6f,0x1.2cec520000000p-6f,0x1.18fb6e0000000p-6f,0x1.07ee0c0000000p-6f,0x1.f262320000000p-7f,0x1.d8aba80000000p-7f,
0x1.c20ed80000000p-7f,0x1.ae077a0000000p-7f,0x1.9c2cc60000000p-7f,0x1.8c2b1c0000000p-7f,0x1.7dbe720000000p-7f,0x1.70af400000000p-7f,0x1.64cfa20000000p-7f,0x1.59f9300000000p-7f,
0x1.500bc60000000p-7f,0x1.46ec460000000p-7f,0x1.3e83680000000p-7f,0x1.36bd5a0000000p-7f,0x1.2f89040000000p-7f,0x1.28d7940000000p-7f,0x1.229c220000000p-7f,0x1.1ccb680000000p-7f,
0x1.175b800000000p-7f,0x1.1243b20000000p-7f,0x1.0d7c440000000p-7f,0x1.08fe580000000p-7f,0x1.04c3d80000000p-7f,0x1.00c74c0000000p-7f,0x1.fa07980000000p-8f,0x1.f2e9f20000000p-8f,
0x1.ec2da60000000p-8f,0x1.e5cb980000000p-8f,0x1.dfbd480000000p-8f,0x1.d9fcd00000000p-8f,0x1.264d4a0000000p-3f,0x1.6b6b460000000p-4f,0x1.05d5620000000p-4f,0x1.98ee540000000p-5f,
0x1.4f9bea0000000p-5f,0x1.1ce8740000000p-5f,0x1.efbe7e0000000p-6f,0x1.b7676a0000000p-6f,0x1.8b32e40000000p-6f,0x1.67a6820000000p-6f,0x1.4a7c500000000p-6f,0x1.32287c0000000p-6f,
0x1.1d94900000000p-6f,0x1.0bf70e0000000p-6f,0x1.f975a20000000p-7f,0x1.dede1c0000000p-7f,0x1.c779e80000000p-7f,0x1.b2c1320000000p-7f,0x1.a048740000000p-7f,0x1.8fb9460000000p-7f,
0x1.80cd9a0000000p-7f,0x1.734be80000000p-7f,0x1.6704b20000000p-7f,0x1.5bd0360000000p-7f,0x1.518d3c0000000p-7f,0x1.481f7e0000000p-7f,0x1.3f6f000000000p-7f,0x1.3767160000000p-7f,
0x1.2ff6080000000p-7f,0x1.290c740000000p-7f,0x1.229cf40000000p-7f,0x1.1c9bda0000000p-7f,0x1.16fedc0000000p-7f,0x1.11bcec0000000p-7f,0x1.0cce080000000p-7f,0x1.082b140000000p-7f,
0x1.03cdbc0000000p-7f,0x1.ff60a20000000p-8f,0x1.f79b900000000p-8f,0x1.f043160000000p-8f,0x1.e94f060000000p-8f,0x1.e2b7f40000000p-8f,0x1.dc77280000000p-8f,0x1.07f5e80000000p-3f,
0x1.4e3ee60000000p-4f,0x1.e8a1c40000000p-5f,0x1.8155a00000000p-5f,0x1.3e8ea00000000p-5f,0x1.0ffbdc0000000p-5f,0x1.db776a0000000p-6f,0x1.a713ca0000000p-6f,0x1.7dc8ae0000000p-6f,
0x1.5c72880000000p-6f,0x1.41013c0000000p-6f,0x1.2a0bd80000000p-6f,0x1.16934a0000000p-6f,0x1.05de660000000p-6f,0x1.eec71e0000000p-7f,0x1.d574bc0000000p-7f,0x1.bf24980000000p-7f,
0x1.ab58880000000p-7f,0x1.99ac0e0000000p-7f,0x1.89ce8a0000000p-7f,0x1.7b7e660000000p-7f,0x1.6e85b00000000p-7f,0x1.62b7c60000000p-7f,0x1.57ef400000000p-7f,0x1.4e0cd40000000p-7f,
0x1.44f5bc0000000p-7f,0x1.3c93500000000p-7f,0x1.34d2140000000p-7f,0x1.2da12e0000000p-7f,0x1.26f20c0000000p-7f,0x1.20b7f60000000p-7f,0x1.1ae7ca0000000p-7f,0x1.1577c40000000p-7f,
0x1.105f420000000p-7f,0x1.0b96a40000000p-7f,0x1.07171c0000000p-7f,0x1.02daa00000000p-7f,0x1.fdb7880000000p-8f,0x1.f62b620000000p-8f,0x1.ef08100000000p-8f,0x1.e845b60000000p-8f,
0x1.e1dd320000000p-8f,0x1.dbc8160000000p-8f,0x1.5217de0000000p-4f,0x1.dbdb180000000p-5f,0x1.6fff8c0000000p-5f,0x1.2d0eec0000000p-5f,0x1.ff1e780000000p-6f,0x1.bd53120000000p-6f,
0x1.8b95600000000p-6f,0x1.64b1100000000p-6f,0x1.4578f00000000p-6f,0x1.2be09a0000000p-6f,0x1.1686ca0000000p-6f,0x1.0474c40000000p-6f,0x1.e9f2620000000p-7f,0x1.cf22680000000p-7f,
0x1.b7b4360000000p-7f,0x1.a3107c0000000p-7f,0x1.90c0d80000000p-7f,0x1.8067f00000000p-7f,0x1.71ba820000000p-7f,0x1.647baa0000000p-7f,0x1.5879940000000p-7f,0x1.4d8ace0000000p-7f,
0x1.438d0a0000000p-7f,0x1.3a63380000000p-7f,0x1.31f4ce0000000p-7f,0x1.2a2ce20000000p-7f,0x1.22f9840000000p-7f,0x1.1c4b3a0000000p-7f,0x1.1614960000000p-7f,0x1.1049ee0000000p-7f,
0x1.0ae1040000000p-7f,0x1.05d0dc0000000p-7f,0x1.01118c0000000p-7f,0x1.f938140000000p-8f,0x1.f0d4320000000p-8f,0x1.e8ec560000000p-8f,0x1.e1768e0000000p-8f,0x1.da69e60000000p-8f,
0x1.d3be5c0000000p-8f,0x1.cd6cb00000000p-8f,0x1.c76e540000000p-8f,0x1.c1bd5c0000000p-8f,0x1.bc545e0000000p-8f,0x1.1b0bda0000000p-3f,0x1.6206ce0000000p-4f,0x1.01595a0000000p-4f,
0x1.9441e60000000p-5f,0x1.4ce9f80000000p-5f,0x1.1b14fe0000000p-5f,0x1.ecad320000000p-6f,0x1.b449fe0000000p-6f,0x1.87ae540000000p-6f,0x1.6387380000000p-6f,0x1.45a5d20000000p-6f,
0x1.2c8c040000000p-6f,0x1.172bc00000000p-6f,0x1.04c0e40000000p-6f,0x1.e972e00000000p-7f,0x1.cd4d260000000p-7f,0x1.b465620000000p-7f,0x1.9e35200000000p-7f,0x1.8a517e0000000p-7f,
0x1.78642a0000000p-7f,0x1.68271c0000000p-7f,0x1.59606e0000000p-7f,0x1.4be0680000000p-7f,0x1.3f7ebc0000000p-7f,0x1.3419b20000000p-7f,0x1.29948a0000000p-7f,0x1.1fd6bc0000000p-7f,
0x1.16cb2c0000000p-7f,0x1.0e5fa20000000p-7f,0x1.0684440000000p-7f,0x1.fe56880000000p-8f,0x1.f091000000000p-8f,0x1.e3a29a0000000p-8f,0x1.d7787c0000000p-8f,0x1.cc01ee0000000p-8f,
0x1.c130100000000p-8f,0x1.b6f58e0000000p-8f,0x1.ad46860000000p-8f,0x1.a418460000000p-8f,0x1.9b61260000000p-8f,0x1.93187a0000000p-8f,0x1.8b365e0000000p-8f,0x1.83b3aa0000000p-8f
};
extern "C" __device__ const float mpas_static_absice3[736] = {
0x1.97b8100000000p-9f,0x1.7e447e0000000p-5f,0x1.0e999e0000000p-4f,0x1.0b899c0000000p-4f,0x1.ec8d520000000p-5f,0x1.bd70840000000p-5f,0x1.91eee60000000p-5f,0x1.6bd1960000000p-5f,
0x1.4b01360000000p-5f,0x1.2ee2b60000000p-5f,0x1.16c6a60000000p-5f,0x1.020bd80000000p-5f,0x1.e0500c0000000p-6f,0x1.c14f200000000p-6f,0x1.a655740000000p-6f,0x1.8ec4aa0000000p-6f,
0x1.7a194c0000000p-6f,0x1.67e5ee0000000p-6f,0x1.57ced40000000p-6f,0x1.4986860000000p-6f,0x1.3ccb6e0000000p-6f,0x1.31657e0000000p-6f,0x1.27247c0000000p-6f,0x1.1ddeb80000000p-6f,
0x1.1570140000000p-6f,0x1.0db8d00000000p-6f,0x1.069d120000000p-6f,0x1.0004320000000p-6f,0x1.f3b0a60000000p-7f,0x1.e80bd80000000p-7f,0x1.dcf6e60000000p-7f,0x1.d2521c0000000p-7f,
0x1.c800e00000000p-7f,0x1.bde9300000000p-7f,0x1.b3f3220000000p-7f,0x1.aa08fe0000000p-7f,0x1.a016a00000000p-7f,0x1.9609820000000p-7f,0x1.8bd0580000000p-7f,0x1.815b340000000p-7f,
0x1.769ae60000000p-7f,0x1.6b81ae0000000p-7f,0x1.6002080000000p-7f,0x1.540fcc0000000p-7f,0x1.479ef60000000p-7f,0x1.3aa4780000000p-7f,0x1.a1da9e0000000p-12f,0x1.136e520000000p-6f,
0x1.ae84e40000000p-6f,0x1.c56be60000000p-6f,0x1.ba7c7e0000000p-6f,0x1.a692a60000000p-6f,0x1.9130f60000000p-6f,0x1.7cbf020000000p-6f,0x1.69f0340000000p-6f,0x1.58dd260000000p-6f,
0x1.4968740000000p-6f,0x1.3b647a0000000p-6f,0x1.2ea22a0000000p-6f,0x1.22f5ee0000000p-6f,0x1.18399e0000000p-6f,0x1.0e4c540000000p-6f,0x1.0512180000000p-6f,0x1.f8e61a0000000p-7f,
0x1.e8b6220000000p-7f,0x1.d971b60000000p-7f,0x1.cafb920000000p-7f,0x1.bd3a740000000p-7f,0x1.b018b80000000p-7f,0x1.a383d40000000p-7f,0x1.976b6c0000000p-7f,0x1.8bc16a0000000p-7f,
0x1.8079a20000000p-7f,0x1.75892e0000000p-7f,0x1.6ae6be0000000p-7f,0x1.6089c40000000p-7f,0x1.566adc0000000p-7f,0x1.4c83a60000000p-7f,0x1.42ce1a0000000p-7f,0x1.39451c0000000p-7f,
0x1.2fe3fa0000000p-7f,0x1.26a6880000000p-7f,0x1.1d89080000000p-7f,0x1.1488160000000p-7f,0x1.0ba0aa0000000p-7f,0x1.02d0000000000p-7f,0x1.f427400000000p-8f,0x1.e2d28c0000000p-8f,
0x1.d19dca0000000p-8f,0x1.c0854a0000000p-8f,0x1.af85a60000000p-8f,0x1.9e9bc80000000p-8f,0x1.1bfb7c0000000p-4f,0x1.5dd1600000000p-4f,0x1.3b77640000000p-4f,0x1.1558bc0000000p-4f,
0x1.ea73820000000p-5f,0x1.b61e340000000p-5f,0x1.8b42ca0000000p-5f,0x1.67acc80000000p-5f,0x1.49ae040000000p-5f,0x1.3009c80000000p-5f,0x1.19d6f60000000p-5f,0x1.0667fa0000000p-5f,
0x1.ea73920000000p-6f,0x1.cbd00e0000000p-6f,0x1.b049340000000p-6f,0x1.97640a0000000p-6f,0x1.80be560000000p-6f,0x1.6c08d20000000p-6f,0x1.5902d00000000p-6f,0x1.4777020000000p-6f,
0x1.3738fc0000000p-6f,0x1.2823a00000000p-6f,0x1.1a176e0000000p-6f,0x1.0cf9980000000p-6f,0x1.00b3380000000p-6f,0x1.ea60c60000000p-7f,0x1.d4bfca0000000p-7f,0x1.c0658c0000000p-7f,
0x1.ad37700000000p-7f,0x1.9b1e540000000p-7f,0x1.8a05520000000p-7f,0x1.79da360000000p-7f,0x1.6a8c500000000p-7f,0x1.5c0cf20000000p-7f,0x1.4e4e980000000p-7f,0x1.41451c0000000p-7f,
0x1.34e57c0000000p-7f,0x1.29259e0000000p-7f,0x1.1dfc3c0000000p-7f,0x1.1360d20000000p-7f,0x1.094b840000000p-7f,0x1.ff6a140000000p-8f,0x1.ed2d440000000p-8f,0x1.dbd3fc0000000p-8f,
0x1.cb526c0000000p-8f,0x1.bb9d940000000p-8f,0x1.699f5e0000000p-3f,0x1.1b2d4e0000000p-3f,0x1.c0909a0000000p-4f,0x1.7020820000000p-4f,0x1.36ef580000000p-4f,0x1.0c75720000000p-4f,
0x1.d77d960000000p-5f,0x1.a399de0000000p-5f,0x1.7974960000000p-5f,0x1.568be20000000p-5f,0x1.3927c40000000p-5f,0x1.2010ec0000000p-5f,0x1.0a653a0000000p-5f,0x1.eef8fc0000000p-6f,
0x1.cdae4a0000000p-6f,0x1.b025420000000p-6f,0x1.95c3600000000p-6f,0x1.7e0dc20000000p-6f,0x1.68a15e0000000p-6f,0x1.552dae0000000p-6f,0x1.4370620000000p-6f,0x1.33327c0000000p-6f,
0x1.2445f00000000p-6f,0x1.1683e60000000p-6f,0x1.09cb5a0000000p-6f,0x1.fc00420000000p-7f,0x1.e613c60000000p-7f,0x1.d1a77e0000000p-7f,0x1.be96aa0000000p-7f,0x1.acc1940000000p-7f,
0x1.9c0c100000000p-7f,0x1.8c5d860000000p-7f,0x1.7da0140000000p-7f,0x1.6fc0420000000p-7f,0x1.62acb20000000p-7f,0x1.5655e20000000p-7f,0x1.4aadfa0000000p-7f,0x1.3fa8440000000p-7f,
0x1.35397a0000000p-7f,0x1.2b574e0000000p-7f,0x1.21f86a0000000p-7f,0x1.1914460000000p-7f,0x1.10a31c0000000p-7f,0x1.089dca0000000p-7f,0x1.00fdcc0000000p-7f,0x1.f37a2e0000000p-8f,
0x1.df2a400000000p-3f,0x1.5a8c080000000p-3f,0x1.0887fa0000000p-3f,0x1.a775aa0000000p-4f,0x1.5ebd080000000p-4f,0x1.29ec720000000p-4f,0x1.01fa100000000p-4f,0x1.c5979e0000000p-5f,
0x1.93af300000000p-5f,0x1.6aecfe0000000p-5f,0x1.4913e00000000p-5f,0x1.2c92040000000p-5f,0x1.1445820000000p-5f,0x1.feb03e0000000p-6f,0x1.da50220000000p-6f,0x1.ba71f60000000p-6f,
0x1.9e53520000000p-6f,0x1.855a480000000p-6f,0x1.6f0b260000000p-6f,0x1.5b01a40000000p-6f,0x1.48eb760000000p-6f,0x1.3884800000000p-6f,0x1.2993ee0000000p-6f,0x1.1bea000000000p-6f,
0x1.0f5e420000000p-6f,0x1.03ce200000000p-6f,0x1.f2382c0000000p-7f,0x1.de5d420000000p-7f,0x1.cbdf520000000p-7f,0x1.ba98020000000p-7f,0x1.aa659a0000000p-7f,0x1.9b2aa80000000p-7f,
0x1.8ccd180000000p-7f,0x1.7f36160000000p-7f,0x1.72512c0000000p-7f,0x1.660c380000000p-7f,0x1.5a56fa0000000p-7f,0x1.4f23140000000p-7f,0x1.4463880000000p-7f,0x1.3a0c980000000p-7f,
0x1.3013ce0000000p-7f,0x1.266faa0000000p-7f,0x1.1d17980000000p-7f,0x1.1403d00000000p-7f,0x1.0b2d460000000p-7f,0x1.028d920000000p-7f,0x1.d524a40000000p-4f,0x1.e11a2c0000000p-4f,
0x1.963c5c0000000p-4f,0x1.5551bc0000000p-4f,0x1.22fd980000000p-4f,0x1.f815860000000p-5f,0x1.bad0f60000000p-5f,0x1.89bb040000000p-5f,0x1.61b94e0000000p-5f,0x1.409e3a0000000p-5f,
0x1.24db6c0000000p-5f,0x1.0d4e8e0000000p-5f,0x1.f23f980000000p-6f,0x1.cf57a20000000p-6f,0x1.b0e9c20000000p-6f,0x1.9631700000000p-6f,0x1.7e93240000000p-6f,0x1.69920a0000000p-6f,
0x1.56c9040000000p-6f,0x1.45e53c0000000p-6f,0x1.36a23e0000000p-6f,0x1.28c6fe0000000p-6f,0x1.1c23820000000p-6f,0x1.108f500000000p-6f,0x1.05e7d60000000p-6f,0x1.f81f0c0000000p-7f,
0x1.e5d9e40000000p-7f,0x1.d4d4020000000p-7f,0x1.c4e7260000000p-7f,0x1.b5f1aa0000000p-7f,0x1.a7d6320000000p-7f,0x1.9a7ad60000000p-7f,0x1.8dc8960000000p-7f,0x1.81ab3c0000000p-7f,
0x1.7610920000000p-7f,0x1.6ae8820000000p-7f,0x1.6024980000000p-7f,0x1.55b7ac0000000p-7f,0x1.4b96400000000p-7f,0x1.41b5aa0000000p-7f,0x1.380c660000000p-7f,0x1.2e91ca0000000p-7f,
0x1.253e000000000p-7f,0x1.1c09e40000000p-7f,0x1.12eef20000000p-7f,0x1.09e73a0000000p-7f,0x1.9089ba0000000p-7f,0x1.b5cca00000000p-5f,0x1.c47b740000000p-5f,0x1.a425e00000000p-5f,
0x1.7f223a0000000p-5f,0x1.5d7ba60000000p-5f,0x1.405a680000000p-5f,0x1.275c600000000p-5f,0x1.11d8800000000p-5f,0x1.fe5ec00000000p-6f,0x1.ddb8820000000p-6f,0x1.c0e9340000000p-6f,
0x1.a747ce0000000p-6f,0x1.904e000000000p-6f,0x1.7b90a40000000p-6f,0x1.68b9720000000p-6f,0x1.5782940000000p-6f,0x1.47b2c80000000p-6f,0x1.391afe0000000p-6f,0x1.2b940e0000000p-6f,
0x1.1efd260000000p-6f,0x1.133ab00000000p-6f,0x1.0835440000000p-6f,0x1.fbb1bc0000000p-7f,0x1.e8289e0000000p-7f,0x1.d5b1a20000000p-7f,0x1.c432f60000000p-7f,0x1.b396300000000p-7f,
0x1.a3c7c60000000p-7f,0x1.94b65a0000000p-7f,0x1.8652c00000000p-7f,0x1.788f740000000p-7f,0x1.6b604c0000000p-7f,0x1.5ebab60000000p-7f,0x1.52950e0000000p-7f,0x1.46e69a0000000p-7f,
0x1.3ba7a40000000p-7f,0x1.30d1200000000p-7f,0x1.265cae0000000p-7f,0x1.1c44860000000p-7f,0x1.1283620000000p-7f,0x1.0914700000000p-7f,0x1.ffe69a0000000p-8f,0x1.ee37d60000000p-8f,
0x1.dd15340000000p-8f,0x1.cc77d80000000p-8f,0x1.12e28c0000000p-4f,0x1.1b5e900000000p-4f,0x1.f5f6a80000000p-5f,0x1.bb58b20000000p-5f,0x1.8c3d300000000p-5f,0x1.666c8c0000000p-5f,
0x1.4785c00000000p-5f,0x1.2dc26c0000000p-5f,0x1.17e15e0000000p-5f,0x1.04ff540000000p-5f,0x1.e8f2320000000p-6f,0x1.cbae5e0000000p-6f,0x1.b180e20000000p-6f,0x1.99e24a0000000p-6f,
0x1.8469e80000000p-6f,0x1.70c5ba0000000p-6f,0x1.5eb46c0000000p-6f,0x1.4e01760000000p-6f,0x1.3e82060000000p-6f,0x1.3012d80000000p-6f,0x1.22968e0000000p-6f,0x1.15f4880000000p-6f,
0x1.0a17f20000000p-6f,0x1.fddde60000000p-7f,0x1.e8d4aa0000000p-7f,0x1.d4f9820000000p-7f,0x1.c235700000000p-7f,0x1.b074120000000p-7f,0x1.9fa39e0000000p-7f,0x1.8fb4280000000p-7f,
0x1.8097960000000p-7f,0x1.7240fa0000000p-7f,0x1.64a5060000000p-7f,0x1.57b93a0000000p-7f,0x1.4b741c0000000p-7f,0x1.3fccfc0000000p-7f,0x1.34bbe60000000p-7f,0x1.2a39780000000p-7f,
0x1.203ee20000000p-7f,0x1.16c5c80000000p-7f,0x1.0dc83c0000000p-7f,0x1.0540ae0000000p-7f,0x1.fa53cc0000000p-8f,0x1.eafde60000000p-8f,0x1.dc765a0000000p-8f,0x1.ceb4440000000p-8f,
0x1.fa95c80000000p-4f,0x1.79bfba0000000p-4f,0x1.2e72d20000000p-4f,0x1.fc3cd20000000p-5f,0x1.b8d05c0000000p-5f,0x1.86d05e0000000p-5f,0x1.60011c0000000p-5f,0x1.40cf380000000p-5f,
0x1.2709c40000000p-5f,0x1.11466e0000000p-5f,0x1.fd21b40000000p-6f,0x1.dc7cb80000000p-6f,0x1.bfa8360000000p-6f,0x1.a5eee80000000p-6f,0x1.8ec8040000000p-6f,0x1.79ca720000000p-6f,
0x1.66a3e40000000p-6f,0x1.5513380000000p-6f,0x1.44e3f60000000p-6f,0x1.35eb6a0000000p-6f,0x1.28066e0000000p-6f,0x1.1b17ea0000000p-6f,0x1.0f07500000000p-6f,0x1.03bff40000000p-6f,
0x1.f2605a0000000p-7f,0x1.de919c0000000p-7f,0x1.cbf9680000000p-7f,0x1.ba811a0000000p-7f,0x1.aa147c0000000p-7f,0x1.9aa2180000000p-7f,0x1.8c1a2a0000000p-7f,0x1.7e6ef00000000p-7f,
0x1.7193ec0000000p-7f,0x1.657df40000000p-7f,0x1.5a22cc0000000p-7f,0x1.4f79640000000p-7f,0x1.45793c0000000p-7f,0x1.3c1ac00000000p-7f,0x1.3356da0000000p-7f,0x1.2b27080000000p-7f,
0x1.2385400000000p-7f,0x1.1c6be80000000p-7f,0x1.15d5c40000000p-7f,0x1.0fbdec0000000p-7f,0x1.0a1fc40000000p-7f,0x1.04f6fc0000000p-7f,0x1.5324500000000p-3f,0x1.d092040000000p-4f,
0x1.64f0860000000p-4f,0x1.2403820000000p-4f,0x1.f0bdd20000000p-5f,0x1.b1b37c0000000p-5f,0x1.81c95e0000000p-5f,0x1.5bf4080000000p-5f,0x1.3d2f760000000p-5f,0x1.2391f60000000p-5f,
0x1.0dd5f20000000p-5f,0x1.f634e60000000p-6f,0x1.d57e360000000p-6f,0x1.b89cea0000000p-6f,0x1.9ee10c0000000p-6f,0x1.87c41a0000000p-6f,0x1.72dd820000000p-6f,0x1.5fdabc0000000p-6f,
0x1.4e79b60000000p-6f,0x1.3e85000000000p-6f,0x1.2fd0be0000000p-6f,0x1.2238b00000000p-6f,0x1.159ea20000000p-6f,0x1.09e9180000000p-6f,0x1.fe04fe0000000p-7f,0x1.e9b09e0000000p-7f,
0x1.d6b59a0000000p-7f,0x1.c4f8420000000p-7f,0x1.b460b00000000p-7f,0x1.a4da2a0000000p-7f,0x1.9652120000000p-7f,0x1.88b84a0000000p-7f,0x1.7bfe3e0000000p-7f,0x1.70172a0000000p-7f,
0x1.64f7660000000p-7f,0x1.5a948c0000000p-7f,0x1.50e54a0000000p-7f,0x1.47e1000000000p-7f,0x1.3f80160000000p-7f,0x1.37bb5e0000000p-7f,0x1.308c6a0000000p-7f,0x1.29ed400000000p-7f,
0x1.23d8660000000p-7f,0x1.1e48c40000000p-7f,0x1.1939aa0000000p-7f,0x1.14a6b60000000p-7f,0x1.6ba55e0000000p-3f,0x1.e358ca0000000p-4f,0x1.6dc1b80000000p-4f,0x1.2848be0000000p-4f,
0x1.f465080000000p-5f,0x1.b279700000000p-5f,0x1.80c5b20000000p-5f,0x1.59cd1c0000000p-5f,0x1.3a4edc0000000p-5f,0x1.203de40000000p-5f,0x1.0a3ec20000000p-5f,0x1.eec4a40000000p-6f,
0x1.cdfd380000000p-6f,0x1.b12f020000000p-6f,0x1.97a0500000000p-6f,0x1.80c3a00000000p-6f,0x1.6c2b020000000p-6f,0x1.597fe40000000p-6f,0x1.487d1c0000000p-6f,0x1.38eaae0000000p-6f,
0x1.2a9ad40000000p-6f,0x1.1d67ba0000000p-6f,0x1.1131e20000000p-6f,0x1.05ded20000000p-6f,0x1.f6b04c0000000p-7f,0x1.e3154e0000000p-7f,0x1.d0cbe00000000p-7f,0x1.bfb7a60000000p-7f,
0x1.afc00c0000000p-7f,0x1.a0cf6e0000000p-7f,0x1.92d2ec0000000p-7f,0x1.85b9e80000000p-7f,0x1.7975a20000000p-7f,0x1.6df8ec0000000p-7f,0x1.6338080000000p-7f,0x1.5928660000000p-7f,
0x1.4fc0b20000000p-7f,0x1.46f8440000000p-7f,0x1.3ec7600000000p-7f,0x1.3726fa0000000p-7f,0x1.3010a20000000p-7f,0x1.297e7a0000000p-7f,0x1.236b1a0000000p-7f,0x1.1dd1880000000p-7f,
0x1.18ad320000000p-7f,0x1.13f9d20000000p-7f,0x1.83b4680000000p-4f,0x1.0774e60000000p-4f,0x1.9e984c0000000p-5f,0x1.5d936a0000000p-5f,0x1.3288220000000p-5f,0x1.13813c0000000p-5f,
0x1.f7913c0000000p-6f,0x1.d1b31c0000000p-6f,0x1.b278180000000p-6f,0x1.980be60000000p-6f,0x1.813b8a0000000p-6f,0x1.6d35620000000p-6f,0x1.5b654c0000000p-6f,0x1.4b5f840000000p-6f,
0x1.3cd3e60000000p-6f,0x1.2f856a0000000p-6f,0x1.2344c80000000p-6f,0x1.17ecdc0000000p-6f,0x1.0d5ffa0000000p-6f,0x1.0386280000000p-6f,0x1.f4976c0000000p-7f,0x1.e341020000000p-7f,
0x1.d2ede40000000p-7f,0x1.c3876a0000000p-7f,0x1.b4fa320000000p-7f,0x1.a735ba0000000p-7f,0x1.9a2bd00000000p-7f,0x1.8dd0220000000p-7f,0x1.8218080000000p-7f,0x1.76f9f20000000p-7f,
0x1.6c6dbc0000000p-7f,0x1.626bf00000000p-7f,0x1.58ee020000000p-7f,0x1.4fee2a0000000p-7f,0x1.47671c0000000p-7f,0x1.3f54140000000p-7f,0x1.37b0c20000000p-7f,0x1.30793e0000000p-7f,
0x1.29a9f20000000p-7f,0x1.233f960000000p-7f,0x1.1d37200000000p-7f,0x1.178dc80000000p-7f,0x1.1240f60000000p-7f,0x1.0d4e420000000p-7f,0x1.08b3720000000p-7f,0x1.046e640000000p-7f,
0x1.3f8d3c0000000p-3f,0x1.9801ac0000000p-4f,0x1.3352200000000p-4f,0x1.f4f14e0000000p-5f,0x1.ab35780000000p-5f,0x1.770abe0000000p-5f,0x1.4fd9980000000p-5f,0x1.3115960000000p-5f,
0x1.181cd20000000p-5f,0x1.034e540000000p-5f,0x1.e3348e0000000p-6f,0x1.c48ffc0000000p-6f,0x1.a9ab140000000p-6f,0x1.91c9d60000000p-6f,0x1.7c60ba0000000p-6f,0x1.6906280000000p-6f,
0x1.5768b20000000p-6f,0x1.47488a0000000p-6f,0x1.3872e40000000p-6f,0x1.2abefc0000000p-6f,0x1.1e0b7a0000000p-6f,0x1.123d0e0000000p-6f,0x1.073cf60000000p-6f,0x1.f9f05c0000000p-7f,
0x1.e6bd4e0000000p-7f,0x1.d4c53e0000000p-7f,0x1.c3f0f00000000p-7f,0x1.b42c180000000p-7f,0x1.a5652a0000000p-7f,0x1.978cfa0000000p-7f,0x1.8a96320000000p-7f,0x1.7e74d00000000p-7f,
0x1.731e9c0000000p-7f,0x1.688a420000000p-7f,0x1.5eaf780000000p-7f,0x1.5586f00000000p-7f,0x1.4d09ca0000000p-7f,0x1.4532320000000p-7f,0x1.3dfaa80000000p-7f,0x1.375e380000000p-7f,
0x1.3158680000000p-7f,0x1.2be5180000000p-7f,0x1.2700880000000p-7f,0x1.22a7460000000p-7f,0x1.1ed6240000000p-7f,0x1.1b8a360000000p-7f,0x1.3f652e0000000p-3f,0x1.955e6c0000000p-4f,
0x1.30caf60000000p-4f,0x1.f08e380000000p-5f,0x1.a76c200000000p-5f,0x1.73bd000000000p-5f,0x1.4cee7a0000000p-5f,0x1.2e7a8c0000000p-5f,0x1.15c46e0000000p-5f,0x1.012eca0000000p-5f,
0x1.df58d60000000p-6f,0x1.c10cc60000000p-6f,0x1.a6781a0000000p-6f,0x1.8ee0840000000p-6f,0x1.79bbee0000000p-6f,0x1.66a18a0000000p-6f,0x1.5540ba0000000p-6f,0x1.455a080000000p-6f,
0x1.36bb320000000p-6f,0x1.293b980000000p-6f,0x1.1cba400000000p-6f,0x1.111bfc0000000p-6f,0x1.064a3c0000000p-6f,0x1.f864140000000p-7f,0x1.e587020000000p-7f,0x1.d3e2000000000p-7f,
0x1.c35df80000000p-7f,0x1.b3e6cc0000000p-7f,0x1.a56b340000000p-7f,0x1.97dc140000000p-7f,0x1.8b2c580000000p-7f,0x1.7f50420000000p-7f,0x1.743daa0000000p-7f,0x1.69eb820000000p-7f,
0x1.6051a40000000p-7f,0x1.5768f20000000p-7f,0x1.4f2ae00000000p-7f,0x1.4791a60000000p-7f,0x1.4098120000000p-7f,0x1.3a39640000000p-7f,0x1.3471540000000p-7f,0x1.2f3bfc0000000p-7f,
0x1.2a95d20000000p-7f,0x1.267b9c0000000p-7f,0x1.22ea640000000p-7f,0x1.1fdf760000000p-7f,0x1.c3c22c0000000p-4f,0x1.25f4560000000p-4f,0x1.c50b220000000p-5f,0x1.7953fe0000000p-5f,
0x1.482e580000000p-5f,0x1.253e3e0000000p-5f,0x1.0ace560000000p-5f,0x1.ebbe2a0000000p-6f,0x1.c96d9c0000000p-6f,0x1.ac8aee0000000p-6f,0x1.93b49e0000000p-6f,0x1.7dfb180000000p-6f,
0x1.6ab6380000000p-6f,0x1.596c8e0000000p-6f,0x1.49c4240000000p-6f,0x1.3b79080000000p-6f,0x1.2e56ca0000000p-6f,0x1.2234680000000p-6f,0x1.16f1540000000p-6f,0x1.0c73600000000p-6f,
0x1.02a54c0000000p-6f,0x1.f2eb300000000p-7f,0x1.e1ab820000000p-7f,0x1.d173500000000p-7f,0x1.c22e220000000p-7f,0x1.b3cab20000000p-7f,0x1.a63a260000000p-7f,0x1.996fd40000000p-7f,
0x1.8d60a60000000p-7f,0x1.8203100000000p-7f,0x1.774ec20000000p-7f,0x1.6d3c6e0000000p-7f,0x1.63c5a20000000p-7f,0x1.5ae4920000000p-7f,0x1.5294620000000p-7f,0x1.4ad05e0000000p-7f,
0x1.4394a00000000p-7f,0x1.3cdd800000000p-7f,0x1.36a7c60000000p-7f,0x1.30f0800000000p-7f,0x1.2bb50e0000000p-7f,0x1.26f3040000000p-7f,0x1.22a83e0000000p-7f,0x1.1ed2ba0000000p-7f,
0x1.1b70ae0000000p-7f,0x1.18806e0000000p-7f,0x1.59c5da0000000p-3f,0x1.b92b300000000p-4f,0x1.4774100000000p-4f,0x1.064d5c0000000p-4f,0x1.b7e5ea0000000p-5f,0x1.7c3d080000000p-5f,
0x1.4fd1340000000p-5f,0x1.2d5c700000000p-5f,0x1.11c8440000000p-5f,0x1.f64e460000000p-6f,0x1.d069f00000000p-6f,0x1.b028520000000p-6f,0x1.94503a0000000p-6f,0x1.7bfde80000000p-6f,
0x1.6687ea0000000p-6f,0x1.536d700000000p-6f,0x1.424b080000000p-6f,0x1.32d2660000000p-6f,0x1.24c5520000000p-6f,0x1.17f19a0000000p-6f,0x1.0c2e4c0000000p-6f,0x1.0159bc0000000p-6f,
0x1.eeafe20000000p-7f,0x1.dc22d60000000p-7f,0x1.cae4d60000000p-7f,0x1.bad43c0000000p-7f,0x1.abd4440000000p-7f,0x1.9dcc2e0000000p-7f,0x1.90a6800000000p-7f,0x1.8450b20000000p-7f,
0x1.78ba660000000p-7f,0x1.6dd5700000000p-7f,0x1.63953c0000000p-7f,0x1.59ee9e0000000p-7f,0x1.50d79e0000000p-7f,0x1.4847ac0000000p-7f,0x1.4036aa0000000p-7f,0x1.389dac0000000p-7f,
0x1.3176520000000p-7f,0x1.2abae40000000p-7f,0x1.2466320000000p-7f,0x1.1e738e0000000p-7f,0x1.18deb80000000p-7f,0x1.13a3cc0000000p-7f,0x1.0ebf3c0000000p-7f,0x1.0a2dce0000000p-7f
};
extern "C" __device__ const float mpas_static_absice0[2] = {
0x1.47ae140000000p-8f,0x1.0000000000000p+0f
};
extern "C" __device__ const float mpas_static_absliq1[928] = {
0x1.ae0a140000000p-10f,0x1.1ad7a20000000p-4f,0x1.3c37da0000000p-4f,0x1.3eb0e00000000p-4f,0x1.3b32560000000p-4f,0x1.3680240000000p-4f,0x1.31b9b60000000p-4f,0x1.2d1b3c0000000p-4f,
0x1.289de40000000p-4f,0x1.2423100000000p-4f,0x1.1e5eaa0000000p-4f,0x1.10f60c0000000p-4f,0x1.04be760000000p-4f,0x1.f33db00000000p-5f,0x1.dee60a0000000p-5f,0x1.cc2e0c0000000p-5f,
0x1.bad5f40000000p-5f,0x1.aaa6320000000p-5f,0x1.9b706a0000000p-5f,0x1.8d0cc40000000p-5f,0x1.7f5a1e0000000p-5f,0x1.723b860000000p-5f,0x1.65980a0000000p-5f,0x1.5959a40000000p-5f,
0x1.4d6d3c0000000p-5f,0x1.41c1000000000p-5f,0x1.3645000000000p-5f,0x1.2aea240000000p-5f,0x1.21b2020000000p-5f,0x1.1828ec0000000p-5f,0x1.0f2b160000000p-5f,0x1.06abb60000000p-5f,
0x1.fd3e1a0000000p-6f,0x1.edf6120000000p-6f,0x1.df6dba0000000p-6f,0x1.d195f40000000p-6f,0x1.c460160000000p-6f,0x1.b7c0cc0000000p-6f,0x1.abac560000000p-6f,0x1.a019100000000p-6f,
0x1.94fe940000000p-6f,0x1.8a53ac0000000p-6f,0x1.80120a0000000p-6f,0x1.76328c0000000p-6f,0x1.6caf560000000p-6f,0x1.6382860000000p-6f,0x1.5aa7160000000p-6f,0x1.5217fa0000000p-6f,
0x1.49d1700000000p-6f,0x1.41cf420000000p-6f,0x1.3a0d420000000p-6f,0x1.3288ea0000000p-6f,0x1.2b3e060000000p-6f,0x1.242a820000000p-6f,0x1.1d4b000000000p-6f,0x1.169cfa0000000p-6f,
0x1.101e5c0000000p-6f,0x1.09cc9c0000000p-6f,0x1.c181e00000000p-3f,0x1.720c060000000p-3f,0x1.45f0700000000p-3f,0x1.2868ba0000000p-3f,0x1.11d2e00000000p-3f,0x1.fd5bac0000000p-4f,
0x1.da22f60000000p-4f,0x1.b793140000000p-4f,0x1.93ea120000000p-4f,0x1.6daa780000000p-4f,0x1.55fac40000000p-4f,0x1.3bb5680000000p-4f,0x1.24f65a0000000p-4f,0x1.110ba60000000p-4f,
0x1.fee3000000000p-5f,0x1.df86ae0000000p-5f,0x1.c365440000000p-5f,0x1.aa029e0000000p-5f,0x1.92fa440000000p-5f,0x1.7dfaa20000000p-5f,0x1.6ac09e0000000p-5f,0x1.5913980000000p-5f,
0x1.48c5000000000p-5f,0x1.39ad000000000p-5f,0x1.2ba9300000000p-5f,0x1.1e9cd80000000p-5f,0x1.126f020000000p-5f,0x1.0709740000000p-5f,0x1.fa31f60000000p-6f,0x1.e796e80000000p-6f,
0x1.d62a440000000p-6f,0x1.c5cf820000000p-6f,0x1.b66f280000000p-6f,0x1.a7f2f80000000p-6f,0x1.9a48160000000p-6f,0x1.8d5d4c0000000p-6f,0x1.8123820000000p-6f,0x1.758ce00000000p-6f,
0x1.6a8d3c0000000p-6f,0x1.6019460000000p-6f,0x1.5627540000000p-6f,0x1.4cadc40000000p-6f,0x1.43a49a0000000p-6f,0x1.3b044e0000000p-6f,0x1.32c5ba0000000p-6f,0x1.2ae2980000000p-6f,
0x1.2355700000000p-6f,0x1.1c18d00000000p-6f,0x1.1527b00000000p-6f,0x1.0e7d720000000p-6f,0x1.0816ba0000000p-6f,0x1.01eeee0000000p-6f,0x1.f805600000000p-7f,0x1.ec9d4c0000000p-7f,
0x1.e1a09a0000000p-7f,0x1.d708900000000p-7f,0x1.cccf520000000p-7f,0x1.c2f1820000000p-7f,0x1.2e42180000000p-2f,0x1.e0cc780000000p-3f,0x1.9594f20000000p-3f,0x1.607d4e0000000p-3f,
0x1.37774a0000000p-3f,0x1.15d1c40000000p-3f,0x1.f2207a0000000p-4f,0x1.bf7f060000000p-4f,0x1.91ece20000000p-4f,0x1.6838ce0000000p-4f,0x1.4cd3b80000000p-4f,0x1.30f9160000000p-4f,
0x1.1923880000000p-4f,0x1.0485d60000000p-4f,0x1.e50b1c0000000p-5f,0x1.c558d00000000p-5f,0x1.a93c700000000p-5f,0x1.9024700000000p-5f,0x1.799c6e0000000p-5f,0x1.6545d40000000p-5f,
0x1.52d3ac0000000p-5f,0x1.4206340000000p-5f,0x1.32a85e0000000p-5f,0x1.248d480000000p-5f,0x1.178f9e0000000p-5f,0x1.0b8f140000000p-5f,0x1.0070680000000p-5f,0x1.ec37600000000p-6f,
0x1.d8ac6a0000000p-6f,0x1.c799f20000000p-6f,0x1.b79c4e0000000p-6f,0x1.a89abe0000000p-6f,0x1.9a7e300000000p-6f,0x1.8d32840000000p-6f,0x1.80a6840000000p-6f,0x1.74ca420000000p-6f,
0x1.698f7c0000000p-6f,0x1.5eea060000000p-6f,0x1.54ce240000000p-6f,0x1.4b31c20000000p-6f,0x1.420ba80000000p-6f,0x1.3953740000000p-6f,0x1.3101960000000p-6f,0x1.290f580000000p-6f,
0x1.2176060000000p-6f,0x1.1a302c0000000p-6f,0x1.1338560000000p-6f,0x1.0c89e60000000p-6f,0x1.0620400000000p-6f,0x1.ffef3a0000000p-7f,0x1.f416be0000000p-7f,0x1.e8b2060000000p-7f,
0x1.ddba5a0000000p-7f,0x1.d328300000000p-7f,0x1.c8f7560000000p-7f,0x1.bf22c00000000p-7f,0x1.b5a5680000000p-7f,0x1.ac7a460000000p-7f,0x1.3425ae0000000p-2f,0x1.e545840000000p-3f,
0x1.9358f20000000p-3f,0x1.597b300000000p-3f,0x1.2d71f40000000p-3f,0x1.0a361a0000000p-3f,0x1.d9fc2a0000000p-4f,0x1.a836ec0000000p-4f,0x1.7cf0820000000p-4f,0x1.56b1f40000000p-4f,
0x1.3bd5120000000p-4f,0x1.2196880000000p-4f,0x1.0b2cf00000000p-4f,0x1.efa2ba0000000p-5f,0x1.cddda40000000p-5f,0x1.b027e60000000p-5f,0x1.95d18e0000000p-5f,0x1.7e4ffc0000000p-5f,
0x1.6934ac0000000p-5f,0x1.56267e0000000p-5f,0x1.44dc740000000p-5f,0x1.351b000000000p-5f,0x1.26b0380000000p-5f,0x1.1971f60000000p-5f,0x1.0d3d6a0000000p-5f,0x1.01f42c0000000p-5f,
0x1.eefa1e0000000p-6f,0x1.db83b40000000p-6f,0x1.c907460000000p-6f,0x1.b8e87e0000000p-6f,0x1.a9cb400000000p-6f,0x1.9b980c0000000p-6f,0x1.8e39ea0000000p-6f,0x1.819ed20000000p-6f,
0x1.75b6660000000p-6f,0x1.6a718e0000000p-6f,0x1.5fc3b40000000p-6f,0x1.55a1880000000p-6f,0x1.4bffb20000000p-6f,0x1.42d4fc0000000p-6f,0x1.3a196c0000000p-6f,0x1.31c4320000000p-6f,
0x1.29cf720000000p-6f,0x1.2234080000000p-6f,0x1.1aec800000000p-6f,0x1.13f2fc0000000p-6f,0x1.0d42de0000000p-6f,0x1.06d7f60000000p-6f,0x1.00ada40000000p-6f,0x1.f5804a0000000p-7f,
0x1.ea18380000000p-7f,0x1.df1d320000000p-7f,0x1.d488820000000p-7f,0x1.ca544c0000000p-7f,0x1.c07b860000000p-7f,0x1.b6f9fc0000000p-7f,0x1.adcaaa0000000p-7f,0x1.a4ea300000000p-7f,
0x1.0f0b280000000p-2f,0x1.b2367e0000000p-3f,0x1.6c8ffc0000000p-3f,0x1.3a72a80000000p-3f,0x1.13e89a0000000p-3f,0x1.e9ccb80000000p-4f,0x1.b641700000000p-4f,0x1.8a59f60000000p-4f,
0x1.643bb40000000p-4f,0x1.42a1300000000p-4f,0x1.2ab5b60000000p-4f,0x1.12cf100000000p-4f,0x1.fcb4500000000p-5f,0x1.d9472e0000000p-5f,0x1.ba4b560000000p-5f,0x1.9ef4b80000000p-5f,
0x1.86a60c0000000p-5f,0x1.70e2f60000000p-5f,0x1.5d491c0000000p-5f,0x1.4b88940000000p-5f,0x1.3b605e0000000p-5f,0x1.2c9a920000000p-5f,0x1.1f0af20000000p-5f,0x1.128c2a0000000p-5f,
0x1.06fe8c0000000p-5f,0x1.f88e880000000p-6f,0x1.e49e7a0000000p-6f,0x1.d205f20000000p-6f,0x1.c093da0000000p-6f,0x1.b12d340000000p-6f,0x1.a2b7ba0000000p-6f,0x1.951d9e0000000p-6f,
0x1.884b280000000p-6f,0x1.7c2f900000000p-6f,0x1.70bb540000000p-6f,0x1.65e1700000000p-6f,0x1.5b95be0000000p-6f,0x1.51cce60000000p-6f,0x1.487db20000000p-6f,0x1.3f9ee60000000p-6f,
0x1.3728f60000000p-6f,0x1.2f13ea0000000p-6f,0x1.275a4c0000000p-6f,0x1.1ff4fe0000000p-6f,0x1.18df600000000p-6f,0x1.1213940000000p-6f,0x1.0b8e3e0000000p-6f,0x1.0549ec0000000p-6f,
0x1.fe875a0000000p-7f,0x1.f2eea00000000p-7f,0x1.e7c2f20000000p-7f,0x1.dcff4a0000000p-7f,0x1.d29dc80000000p-7f,0x1.c8988a0000000p-7f,0x1.beea8c0000000p-7f,0x1.b58f9a0000000p-7f,
0x1.ac84580000000p-7f,0x1.a3c3c00000000p-7f,0x1.68eea20000000p-4f,0x1.b53a3e0000000p-4f,0x1.914e8c0000000p-4f,0x1.707c840000000p-4f,0x1.5619120000000p-4f,0x1.40440e0000000p-4f,
0x1.2d70cc0000000p-4f,0x1.1c8c280000000p-4f,0x1.0cce7a0000000p-4f,0x1.fb34c00000000p-5f,0x1.dde26a0000000p-5f,0x1.c1fdd20000000p-5f,0x1.a9b0d40000000p-5f,0x1.9421c00000000p-5f,
0x1.80bea20000000p-5f,0x1.6f1f720000000p-5f,0x1.5ef87e0000000p-5f,0x1.500fae0000000p-5f,0x1.4237b20000000p-5f,0x1.354cb40000000p-5f,0x1.2931880000000p-5f,0x1.1dcddc0000000p-5f,
0x1.130d8e0000000p-5f,0x1.08dfd80000000p-5f,0x1.fe6bac0000000p-6f,0x1.ec07240000000p-6f,0x1.da7b760000000p-6f,0x1.c9b62c0000000p-6f,0x1.bb30c40000000p-6f,0x1.ae45fa0000000p-6f,
0x1.a1d53c0000000p-6f,0x1.95e0a80000000p-6f,0x1.8a67ce0000000p-6f,0x1.7f682a0000000p-6f,0x1.74df3a0000000p-6f,0x1.6ac71e0000000p-6f,0x1.611c7a0000000p-6f,0x1.57d9700000000p-6f,
0x1.4ef9ce0000000p-6f,0x1.4677b40000000p-6f,0x1.3e4ef00000000p-6f,0x1.367a7a0000000p-6f,0x1.2ef6220000000p-6f,0x1.27bdb40000000p-6f,0x1.20cd000000000p-6f,0x1.1a203c0000000p-6f,
0x1.13b4120000000p-6f,0x1.0d844e0000000p-6f,0x1.078ed60000000p-6f,0x1.01d0500000000p-6f,0x1.f88b980000000p-7f,0x1.edd8b60000000p-7f,0x1.e384a00000000p-7f,0x1.d9889c0000000p-7f,
0x1.cfe22c0000000p-7f,0x1.c68c400000000p-7f,0x1.bd83840000000p-7f,0x1.b4c3c20000000p-7f,0x1.6209740000000p-5f,0x1.2d7f600000000p-4f,0x1.1e0a420000000p-4f,0x1.107a860000000p-4f,
0x1.06f1200000000p-4f,0x1.fed0200000000p-5f,0x1.f0f5360000000p-5f,0x1.e23f240000000p-5f,0x1.d096860000000p-5f,0x1.b93eb00000000p-5f,0x1.9941500000000p-5f,0x1.8e5e320000000p-5f,
0x1.811f500000000p-5f,0x1.72fa5e0000000p-5f,0x1.64bbd60000000p-5f,0x1.56d19e0000000p-5f,0x1.4974f40000000p-5f,0x1.3cc0100000000p-5f,0x1.30bc600000000p-5f,0x1.2569460000000p-5f,
0x1.1ac0400000000p-5f,0x1.10b8b40000000p-5f,0x1.07485e0000000p-5f,0x1.fcc9140000000p-6f,0x1.ec05e20000000p-6f,0x1.dc33de0000000p-6f,0x1.cd3fba0000000p-6f,0x1.bf18440000000p-6f,
0x1.ae9b1e0000000p-6f,0x1.a302ce0000000p-6f,0x1.97a1de0000000p-6f,0x1.8c8a520000000p-6f,0x1.81c7140000000p-6f,0x1.775edc0000000p-6f,0x1.6d55020000000p-6f,0x1.63a9880000000p-6f,
0x1.5a5c6c0000000p-6f,0x1.516b980000000p-6f,0x1.48d41a0000000p-6f,0x1.4093020000000p-6f,0x1.38a5620000000p-6f,0x1.31070a0000000p-6f,0x1.29b5060000000p-6f,0x1.22ab920000000p-6f,
0x1.1be7520000000p-6f,0x1.1564800000000p-6f,0x1.0f209a0000000p-6f,0x1.0918420000000p-6f,0x1.03488a0000000p-6f,0x1.fb5d040000000p-7f,0x1.f08f4a0000000p-7f,0x1.e623b60000000p-7f,
0x1.dc14680000000p-7f,0x1.d25c580000000p-7f,0x1.c8f9020000000p-7f,0x1.bfe55e0000000p-7f,0x1.b71e0e0000000p-7f,0x1.ae9fbc0000000p-7f,0x1.2292820000000p-3f,0x1.25091e0000000p-4f,
0x1.022f6a0000000p-4f,0x1.f4a3ac0000000p-5f,0x1.ed1a140000000p-5f,0x1.e54f7a0000000p-5f,0x1.da4a640000000p-5f,0x1.c9d4ca0000000p-5f,0x1.b14b320000000p-5f,0x1.8cdf0c0000000p-5f,
0x1.7984f20000000p-5f,0x1.7590dc0000000p-5f,0x1.6ce1400000000p-5f,0x1.61c8de0000000p-5f,0x1.55a0b00000000p-5f,0x1.49320e0000000p-5f,0x1.3cf1220000000p-5f,0x1.311ebc0000000p-5f,
0x1.25dd740000000p-5f,0x1.1b3cd20000000p-5f,0x1.1141700000000p-5f,0x1.07e8cc0000000p-5f,0x1.fe5a0e0000000p-6f,0x1.ee0b760000000p-6f,0x1.ded3ca0000000p-6f,0x1.d09fc00000000p-6f,
0x1.c35c0a0000000p-6f,0x1.b6f7e40000000p-6f,0x1.a633dc0000000p-6f,0x1.9ac0e20000000p-6f,0x1.8fa2a00000000p-6f,0x1.84dfce0000000p-6f,0x1.7a7a860000000p-6f,0x1.7074de0000000p-6f,
0x1.66cd940000000p-6f,0x1.5d82fe0000000p-6f,0x1.5493000000000p-6f,0x1.4bfa3e0000000p-6f,0x1.43b5cc0000000p-6f,0x1.3bc24e0000000p-6f,0x1.341c680000000p-6f,0x1.2cc0520000000p-6f,
0x1.25ab200000000p-6f,0x1.1ed9740000000p-6f,0x1.1847f60000000p-6f,0x1.11f3b20000000p-6f,0x1.0bd9520000000p-6f,0x1.05f6ba0000000p-6f,0x1.0048fc0000000p-6f,0x1.f59a4c0000000p-7f,
0x1.eb03180000000p-7f,0x1.e0c7560000000p-7f,0x1.d6e1fa0000000p-7f,0x1.cd50800000000p-7f,0x1.c40d0a0000000p-7f,0x1.bb15ec0000000p-7f,0x1.b266f00000000p-7f,0x1.a9fbe80000000p-7f,
0x1.138c700000000p-4f,0x1.0ec03c0000000p-4f,0x1.095dac0000000p-4f,0x1.0398a60000000p-4f,0x1.fb12900000000p-5f,0x1.ee40860000000p-5f,0x1.e029a00000000p-5f,0x1.cf90cc0000000p-5f,
0x1.ba3c720000000p-5f,0x1.9c7f280000000p-5f,0x1.82575e0000000p-5f,0x1.7bce420000000p-5f,0x1.70e57c0000000p-5f,0x1.63f6b40000000p-5f,0x1.5652f20000000p-5f,0x1.48b6880000000p-5f,
0x1.3b88360000000p-5f,0x1.2efd440000000p-5f,0x1.232d980000000p-5f,0x1.1820f40000000p-5f,0x1.0dd4d40000000p-5f,0x1.04410e0000000p-5f,0x1.f6b6ac0000000p-6f,0x1.e62fe00000000p-6f,
0x1.d6d5640000000p-6f,0x1.c88f500000000p-6f,0x1.bb47d40000000p-6f,0x1.aee98c0000000p-6f,0x1.9d9d340000000p-6f,0x1.921a4a0000000p-6f,0x1.8703940000000p-6f,0x1.7c58400000000p-6f,
0x1.72170a0000000p-6f,0x1.683d6c0000000p-6f,0x1.5ec7a20000000p-6f,0x1.55b1e40000000p-6f,0x1.4cf8d80000000p-6f,0x1.4497e00000000p-6f,0x1.3c8b360000000p-6f,0x1.34cf140000000p-6f,
0x1.2d5fb40000000p-6f,0x1.2638780000000p-6f,0x1.1f56de0000000p-6f,0x1.18b71c0000000p-6f,0x1.12556e0000000p-6f,0x1.0c2f4e0000000p-6f,0x1.0641ce0000000p-6f,0x1.0089fe0000000p-6f,
0x1.f609dc0000000p-7f,0x1.eb61e40000000p-7f,0x1.e1155a0000000p-7f,0x1.d721ba0000000p-7f,0x1.cd81fe0000000p-7f,0x1.c431f40000000p-7f,0x1.bb2f160000000p-7f,0x1.b2745c0000000p-7f,
0x1.a9ff420000000p-7f,0x1.a1cc6e0000000p-7f,0x1.4677b40000000p-4f,0x1.38deda0000000p-4f,0x1.2dab840000000p-4f,0x1.2442840000000p-4f,0x1.1bded40000000p-4f,0x1.1394ee0000000p-4f,
0x1.0a553e0000000p-4f,0x1.fdddb20000000p-5f,0x1.e0209c0000000p-5f,0x1.b893fa0000000p-5f,0x1.a5a0a40000000p-5f,0x1.9751560000000p-5f,0x1.87145c0000000p-5f,0x1.7657aa0000000p-5f,
0x1.65dfc40000000p-5f,0x1.5612900000000p-5f,0x1.4721260000000p-5f,0x1.391e2e0000000p-5f,0x1.2c0b560000000p-5f,0x1.1fe1460000000p-5f,0x1.14936a0000000p-5f,0x1.0a13140000000p-5f,
0x1.0051280000000p-5f,0x1.ee7d200000000p-6f,0x1.dd9a100000000p-6f,0x1.cddd6e0000000p-6f,0x1.bf2eea0000000p-6f,0x1.b177dc0000000p-6f,0x1.a06c880000000p-6f,0x1.9468d80000000p-6f,
0x1.88dc480000000p-6f,0x1.7dc66a0000000p-6f,0x1.7325fc0000000p-6f,0x1.68f6640000000p-6f,0x1.5f34ae0000000p-6f,0x1.55db6a0000000p-6f,0x1.4ce6ce0000000p-6f,0x1.4450fe0000000p-6f,
0x1.3c16300000000p-6f,0x1.3431600000000p-6f,0x1.2c9e5a0000000p-6f,0x1.2558140000000p-6f,0x1.1e5b340000000p-6f,0x1.17a38a0000000p-6f,0x1.112dba0000000p-6f,0x1.0af6680000000p-6f,
0x1.04f9640000000p-6f,0x1.fe69fe0000000p-7f,0x1.f34a100000000p-7f,0x1.e88fa00000000p-7f,0x1.de34d40000000p-7f,0x1.d434a00000000p-7f,0x1.ca8ad40000000p-7f,0x1.c1333c0000000p-7f,
0x1.b829a80000000p-7f,0x1.af6b920000000p-7f,0x1.a6f3f60000000p-7f,0x1.9ebf740000000p-7f,0x1.320c8c0000000p-3f,0x1.117acc0000000p-3f,0x1.f1d6060000000p-4f,0x1.c9b3080000000p-4f,
0x1.a6f71a0000000p-4f,0x1.877c2a0000000p-4f,0x1.69d5500000000p-4f,0x1.4cfb400000000p-4f,0x1.30243a0000000p-4f,0x1.12ae740000000p-4f,0x1.05a2f20000000p-4f,0x1.e9b53a0000000p-5f,
0x1.ca9f2a0000000p-5f,0x1.ae572c0000000p-5f,0x1.94cafe0000000p-5f,0x1.7dc2380000000p-5f,0x1.68f9be0000000p-5f,0x1.562f4c0000000p-5f,0x1.4526100000000p-5f,0x1.35a7ee0000000p-5f,
0x1.2785140000000p-5f,0x1.1a94a20000000p-5f,0x1.0eb2b60000000p-5f,0x1.03c04a0000000p-5f,0x1.f345060000000p-6f,0x1.e083ce0000000p-6f,0x1.cf14a60000000p-6f,0x1.bed3100000000p-6f,
0x1.acea440000000p-6f,0x1.9ef8120000000p-6f,0x1.91cbda0000000p-6f,0x1.8557580000000p-6f,0x1.798cb40000000p-6f,0x1.6e5eee0000000p-6f,0x1.63c31e0000000p-6f,0x1.59ae5c0000000p-6f,
0x1.5017040000000p-6f,0x1.46f4460000000p-6f,0x1.3e3e940000000p-6f,0x1.35edf80000000p-6f,0x1.2dfc280000000p-6f,0x1.2662d60000000p-6f,0x1.1f1c240000000p-6f,0x1.1823780000000p-6f,
0x1.11735a0000000p-6f,0x1.0b08060000000p-6f,0x1.04dd4a0000000p-6f,0x1.fddde60000000p-7f,0x1.f273500000000p-7f,0x1.e775c60000000p-7f,0x1.dcddbc0000000p-7f,0x1.d2a7020000000p-7f,
0x1.c8cae00000000p-7f,0x1.bf45fc0000000p-7f,0x1.b6134c0000000p-7f,0x1.ad2ea20000000p-7f,0x1.a494a00000000p-7f,0x1.9c41140000000p-7f,0x1.30bae80000000p-5f,0x1.3e54640000000p-5f,
0x1.46eb0c0000000p-5f,0x1.4b3e8e0000000p-5f,0x1.4b74de0000000p-5f,0x1.478b440000000p-5f,0x1.3f73120000000p-5f,0x1.331b4a0000000p-5f,0x1.2272bc0000000p-5f,0x1.0d690a0000000p-5f,
0x1.10723e0000000p-5f,0x1.0825680000000p-5f,0x1.ffcf580000000p-6f,0x1.efc0ee0000000p-6f,0x1.e04a580000000p-6f,0x1.d180900000000p-6f,0x1.c36b900000000p-6f,0x1.b60b540000000p-6f,
0x1.a95bae0000000p-6f,0x1.9d55e60000000p-6f,0x1.91f19a0000000p-6f,0x1.8725220000000p-6f,0x1.7ce81e0000000p-6f,0x1.7331500000000p-6f,0x1.69f8560000000p-6f,0x1.6134620000000p-6f,
0x1.58de540000000p-6f,0x1.50ef060000000p-6f,0x1.4744ce0000000p-6f,0x1.40a4340000000p-6f,0x1.3a21d00000000p-6f,0x1.33c1640000000p-6f,0x1.2d849e0000000p-6f,0x1.276d980000000p-6f,
0x1.217cbc0000000p-6f,0x1.1bb2780000000p-6f,0x1.160e600000000p-6f,0x1.1091480000000p-6f,0x1.0b39840000000p-6f,0x1.0606aa0000000p-6f,0x1.00f84e0000000p-6f,0x1.f81a580000000p-7f,
0x1.ee88e00000000p-7f,0x1.e53a820000000p-7f,0x1.dc2d920000000p-7f,0x1.d35f8e0000000p-7f,0x1.cacdf00000000p-7f,0x1.c277e00000000p-7f,0x1.ba5ada0000000p-7f,0x1.b275320000000p-7f,
0x1.aac4640000000p-7f,0x1.a347980000000p-7f,0x1.9bfc4a0000000p-7f,0x1.94e0ce0000000p-7f,0x1.8df3740000000p-7f,0x1.8733660000000p-7f,0x1.809e200000000p-7f,0x1.7a32cc0000000p-7f,
0x1.fef6ec0000000p-6f,0x1.6f4b480000000p-5f,0x1.91976c0000000p-5f,0x1.96a7e20000000p-5f,0x1.8eca9e0000000p-5f,0x1.80b45a0000000p-5f,0x1.6f84880000000p-5f,0x1.5ccfae0000000p-5f,
0x1.496e740000000p-5f,0x1.35d9d60000000p-5f,0x1.3299440000000p-5f,0x1.273a000000000p-5f,0x1.1c52b20000000p-5f,0x1.11f7e40000000p-5f,0x1.0830f20000000p-5f,0x1.fdf9960000000p-6f,
0x1.ecac660000000p-6f,0x1.dc6ba60000000p-6f,0x1.cd26240000000p-6f,0x1.becb820000000p-6f,0x1.b14a900000000p-6f,0x1.a492f20000000p-6f,0x1.9895f80000000p-6f,0x1.8d45620000000p-6f,
0x1.8293c40000000p-6f,0x1.7876340000000p-6f,0x1.6ee0f40000000p-6f,0x1.65caca0000000p-6f,0x1.5b006c0000000p-6f,0x1.5345f80000000p-6f,0x1.4bba120000000p-6f,0x1.445f400000000p-6f,
0x1.3d372e0000000p-6f,0x1.3642460000000p-6f,0x1.2f808c0000000p-6f,0x1.28f1900000000p-6f,0x1.2294800000000p-6f,0x1.1c68160000000p-6f,0x1.166be80000000p-6f,0x1.109ddc0000000p-6f,
0x1.0afcb40000000p-6f,0x1.0587280000000p-6f,0x1.003bfa0000000p-6f,0x1.f632f80000000p-7f,0x1.ec3cd40000000p-7f,0x1.e2915a0000000p-7f,0x1.d92eda0000000p-7f,0x1.d012d20000000p-7f,
0x1.c73abc0000000p-7f,0x1.bea2680000000p-7f,0x1.b649d40000000p-7f,0x1.ae2bf80000000p-7f,0x1.a648d40000000p-7f,0x1.9e9d0e0000000p-7f,0x1.9726f80000000p-7f,0x1.8fe4e40000000p-7f,
0x1.88d3780000000p-7f,0x1.81f2b40000000p-7f,0x1.047c660000000p-6f,0x1.1f410e0000000p-5f,0x1.48608c0000000p-5f,0x1.4da2b60000000p-5f,0x1.461fd40000000p-5f,0x1.3a01180000000p-5f,
0x1.2c819e0000000p-5f,0x1.1efce60000000p-5f,0x1.12051a0000000p-5f,0x1.05d2de0000000p-5f,0x1.00fe2c0000000p-5f,0x1.f101600000000p-6f,0x1.e0bb2c0000000p-6f,0x1.d142e80000000p-6f,
0x1.c29ee20000000p-6f,0x1.b4cdd20000000p-6f,0x1.a7c75a0000000p-6f,0x1.9b80900000000p-6f,0x1.8feef60000000p-6f,0x1.85058e0000000p-6f,0x1.7ab9040000000p-6f,0x1.70fe040000000p-6f,
0x1.67ca800000000p-6f,0x1.5f14620000000p-6f,0x1.56d2740000000p-6f,0x1.4efc520000000p-6f,0x1.478a6e0000000p-6f,0x1.4076120000000p-6f,0x1.37b6500000000p-6f,0x1.31b8e00000000p-6f,
0x1.2bd2160000000p-6f,0x1.2604e20000000p-6f,0x1.20549e0000000p-6f,0x1.1ac2fa0000000p-6f,0x1.1550ca0000000p-6f,0x1.0ffee60000000p-6f,0x1.0acd4e0000000p-6f,0x1.05bcd80000000p-6f,
0x1.00cc440000000p-6f,0x1.f7f7f40000000p-7f,0x1.ee964c0000000p-7f,0x1.e571e00000000p-7f,0x1.dc8ab00000000p-7f,0x1.d3de3a0000000p-7f,0x1.cb6acc0000000p-7f,0x1.c32f960000000p-7f,
0x1.bb2a0e0000000p-7f,0x1.b3595e0000000p-7f,0x1.abbbdc0000000p-7f,0x1.a450ac0000000p-7f,0x1.9d14780000000p-7f,0x1.9606680000000p-7f,0x1.8f25a20000000p-7f,0x1.88707c0000000p-7f,
0x1.81e5480000000p-7f,0x1.7b82580000000p-7f,0x1.7546d40000000p-7f,0x1.6f31100000000p-7f,0x1.490ae00000000p-8f,0x1.648a580000000p-6f,0x1.a287140000000p-6f,0x1.a9237a0000000p-6f,
0x1.9f94860000000p-6f,0x1.91de4e0000000p-6f,0x1.840b280000000p-6f,0x1.7774400000000p-6f,0x1.6c76b60000000p-6f,0x1.6311480000000p-6f,0x1.5a70fa0000000p-6f,0x1.5135e80000000p-6f,
0x1.486c800000000p-6f,0x1.4018f40000000p-6f,0x1.3839980000000p-6f,0x1.30c9cc0000000p-6f,0x1.29c41e0000000p-6f,0x1.2321d80000000p-6f,0x1.1cdcb00000000p-6f,0x1.16ed820000000p-6f,
0x1.114f480000000p-6f,0x1.0bfb4c0000000p-6f,0x1.06ecee0000000p-6f,0x1.021ebe0000000p-6f,0x1.fb19e80000000p-7f,0x1.f264360000000p-7f,0x1.ea17600000000p-7f,0x1.e22b020000000p-7f,
0x1.d6d2e00000000p-7f,0x1.d06a100000000p-7f,0x1.c9f8dc0000000p-7f,0x1.c388800000000p-7f,0x1.bd1d2c0000000p-7f,0x1.b6bbea0000000p-7f,0x1.b068e80000000p-7f,0x1.aa26b00000000p-7f,
0x1.a3f7c20000000p-7f,0x1.9dddcc0000000p-7f,0x1.97d9a60000000p-7f,0x1.91edd40000000p-7f,0x1.8c1a560000000p-7f,0x1.865e540000000p-7f,0x1.80bc540000000p-7f,0x1.7b337e0000000p-7f,
0x1.75c2fa0000000p-7f,0x1.706c7a0000000p-7f,0x1.6b2e4c0000000p-7f,0x1.66094a0000000p-7f,0x1.60fbc40000000p-7f,0x1.5c07680000000p-7f,0x1.5729b20000000p-7f,0x1.5264500000000p-7f,
0x1.4db4be0000000p-7f,0x1.491bd20000000p-7f,0x1.4498f60000000p-7f,0x1.402b6a0000000p-7f,0x1.3bd2ea0000000p-7f,0x1.378ee20000000p-7f,0x1.ced3020000000p-5f,0x1.7294720000000p-4f,
0x1.4c7ba40000000p-4f,0x1.20f3cc0000000p-4f,0x1.fbf2540000000p-5f,0x1.c612340000000p-5f,0x1.9c783c0000000p-5f,0x1.7bf0060000000p-5f,0x1.6221c80000000p-5f,0x1.4d617c0000000p-5f,
0x1.32f2300000000p-5f,0x1.2125ea0000000p-5f,0x1.1172ba0000000p-5f,0x1.0380520000000p-5f,0x1.ee0f3c0000000p-6f,0x1.d79f8e0000000p-6f,0x1.c3562c0000000p-6f,0x1.b0e7280000000p-6f,
0x1.a0139c0000000p-6f,0x1.90a85a0000000p-6f,0x1.8279580000000p-6f,0x1.7561ac0000000p-6f,0x1.6941e20000000p-6f,0x1.5dff900000000p-6f,0x1.5383340000000p-6f,0x1.49b8b00000000p-6f,
0x1.408f3c0000000p-6f,0x1.37f6e80000000p-6f,0x1.2e1e200000000p-6f,0x1.26f2480000000p-6f,0x1.200e940000000p-6f,0x1.196fa80000000p-6f,0x1.1311c00000000p-6f,0x1.0cf2560000000p-6f,
0x1.070d3a0000000p-6f,0x1.015fe60000000p-6f,0x1.f7cfb00000000p-7f,0x1.ed428e0000000p-7f,0x1.e315e40000000p-7f,0x1.d942fc0000000p-7f,0x1.cfc5a60000000p-7f,0x1.c69a840000000p-7f,
0x1.bdbc8e0000000p-7f,0x1.b5286c0000000p-7f,0x1.acd9e80000000p-7f,0x1.a4ce820000000p-7f,0x1.9d02da0000000p-7f,0x1.9574700000000p-7f,0x1.8e1f120000000p-7f,0x1.8701100000000p-7f,
0x1.8017e80000000p-7f,0x1.7961ec0000000p-7f,0x1.72dbc00000000p-7f,0x1.6c83b60000000p-7f,0x1.6658240000000p-7f,0x1.6056820000000p-7f,0x1.5a7dfa0000000p-7f,0x1.54cc0a0000000p-7f
};
extern "C" __device__ const int mpas_static_ngb[140] = {
1,1,1,1,1,1,1,1,
1,1,2,2,2,2,2,2,
2,2,2,2,2,2,3,3,
3,3,3,3,3,3,3,3,
3,3,3,3,3,3,4,4,
4,4,4,4,4,4,4,4,
4,4,4,4,5,5,5,5,
5,5,5,5,5,5,5,5,
5,5,5,5,6,6,6,6,
6,6,6,6,7,7,7,7,
7,7,7,7,7,7,7,7,
8,8,8,8,8,8,8,8,
9,9,9,9,9,9,9,9,
9,9,9,9,10,10,10,10,
10,10,11,11,11,11,11,11,
11,11,12,12,12,12,12,12,
12,12,13,13,13,13,14,14,
15,15,16,16
};
extern "C" __device__ const int mpas_static_ngs[16] = {
10,22,38,52,68,76,88,96,
108,114,122,130,134,136,138,140
};
extern "C" __device__ const float mpas_static_delwave[16] = {
0x1.5400000000000p+8f,0x1.2c00000000000p+7f,0x1.0400000000000p+7f,0x1.1800000000000p+6f,0x1.e000000000000p+6f,0x1.4000000000000p+7f,0x1.9000000000000p+6f,0x1.9000000000000p+6f,
0x1.a400000000000p+7f,0x1.6800000000000p+6f,0x1.4000000000000p+8f,0x1.1800000000000p+8f,0x1.5400000000000p+7f,0x1.0400000000000p+7f,0x1.b800000000000p+7f,0x1.4500000000000p+9f
};
extern "C" __device__ const int mpas_static_ngc[16] = {
10,12,16,14,16,8,12,8,
12,6,8,8,4,2,2,2
};
extern "C" __device__ const int mpas_static_ngn[140] = {
1,1,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
1,1,2,2,2,2,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,3,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,2,2,2,2,
2,2,2,2,2,2,1,1,
1,1,1,1,1,1,2,2,
2,2,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
4,4,1,1,2,2,2,2,
3,3,1,1,1,1,2,2,
4,4,3,3,4,6,8,8,
8,8,4,12
};
extern "C" __device__ const int mpas_static_ngm[256] = {
1,2,3,3,4,4,5,5,
6,6,7,7,8,8,9,10,
1,2,3,4,5,6,7,8,
9,9,10,10,11,11,12,12,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,15,16,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,14,14,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,15,16,
1,1,2,2,3,3,4,4,
5,5,6,6,7,7,8,8,
1,1,2,2,3,4,5,6,
7,8,9,10,11,11,12,12,
1,1,2,2,3,3,4,4,
5,5,6,6,7,7,8,8,
1,2,3,4,5,6,7,8,
9,9,10,10,11,11,12,12,
1,1,2,2,3,3,4,4,
5,5,5,5,6,6,6,6,
1,2,3,3,4,4,5,5,
6,6,7,7,7,8,8,8,
1,2,3,4,5,5,6,6,
7,7,7,7,8,8,8,8,
1,1,1,2,2,2,3,3,
3,3,4,4,4,4,4,4,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
1,1,1,1,2,2,2,2,
2,2,2,2,2,2,2,2
};
extern "C" __device__ const float mpas_static_wt[16] = {
0x1.38d6ca0000000p-3f,0x1.3181980000000p-3f,0x1.23034e0000000p-3f,0x1.0db2c80000000p-3f,0x1.e41ff00000000p-4f,0x1.a181720000000p-4f,0x1.5519f40000000p-4f,0x1.00b45e0000000p-4f,
0x1.5c193c0000000p-5f,0x1.2f3c000000000p-8f,0x1.f5be000000000p-9f,0x1.8ca0000000000p-9f,0x1.22fa000000000p-9f,0x1.72ac000000000p-10f,0x1.17720c0000000p-11f,0x1.3a92a40000000p-14f
};
extern "C" __device__ const float mpas_static_a0[16] = {
0x1.a8f5c20000000p+0f,0x1.8ccccc0000000p+0f,0x1.947ae20000000p+0f,0x1.a8f5c20000000p+0f,0x1.8a3d700000000p+0f,0x1.7439580000000p+0f,0x1.e3d70a0000000p+0f,0x1.547ae20000000p+0f,
0x1.ab020c0000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f,0x1.a8f5c20000000p+0f
};
extern "C" __device__ const float mpas_static_a1[16] = {
0x0.0p+0f,0x1.0000000000000p-2f,0x1.c28f5c0000000p-3f,0x0.0p+0f,0x1.0a3d700000000p-3f,0x1.c8b43a0000000p-2f,-0x1.99999a0000000p-4f,0x1.99999a0000000p-2f,
-0x1.89374c0000000p-8f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f
};
extern "C" __device__ const float mpas_static_a2[16] = {
0x0.0p+0f,-0x1.8000000000000p+3f,-0x1.7666660000000p+3f,0x0.0p+0f,-0x1.70a3d80000000p-1f,-0x1.f1a9fc0000000p-3f,0x1.851eb80000000p-3f,-0x1.fbe76c0000000p-5f,
0x1.a7ef9e0000000p-2f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f,0x0.0p+0f
};
// MPAS: module_ra_rrtmg_lw.F:8082-8113.
extern "C" __global__ void mpas_init_rwgt(float* rwgt){
 // MPAS: module_ra_rrtmg_lw.F:8082-8113 (rrtmg_lw_ini).
 if(threadIdx.x||blockIdx.x)return;
 int total=0;float sum[16];
 for(int band=0;band<16;++band){
   if(mpas_static_ngc[band]<16){
     int old=0;
     for(int g=0;g<mpas_static_ngc[band];++g){float weight=0.0f;
       for(int j=0;j<mpas_static_ngn[total];++j){weight=RLW_AD(weight,mpas_static_wt[old]);++old;}
       sum[g]=weight;++total;
     }
     for(int g=0;g<16;++g)rwgt[band*16+g]=RLW_DV(mpas_static_wt[g],sum[mpas_static_ngm[band*16+g]-1]);
   } else {for(int g=0;g<16;++g){rwgt[band*16+g]=1.0f;++total;}}
 }
}
// MPAS: module_ra_rrtmg_lw.F:8054-8079.
extern "C" __global__ void mpas_init_tables(float* tau,float* exp_table,float* tfn){
 // MPAS: module_ra_rrtmg_lw.F:8054-8079 (rrtmg_lw_ini).
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>10000)return;
 if(i==0){tau[i]=0.0f;exp_table[i]=1.0f;tfn[i]=0.0f;return;}
 if(i==10000){tau[i]=1.e10f;exp_table[i]=1.e-20f;tfn[i]=1.0f;return;}
 float t=RLW_DV((float)i,10000.0f);
 tau[i]=RLW_DV(RLW_MU((1.0f/hex_divisor_3e8e5604),t),RLW_SU(1.0f,t));
 exp_table[i]=fmaxf(rlw_exp(-tau[i]),1.e-20f);
 if(tau[i]<0.06f)tfn[i]=RLW_DV(tau[i],6.0f);
 else tfn[i]=RLW_SU(1.0f,RLW_MU(2.0f,RLW_SU(RLW_DV(1.0f,tau[i]),RLW_DV(exp_table[i],RLW_SU(1.0f,exp_table[i])))));
}
// MPAS: module_ra_rrtmg_lw.F:8331-9982.
extern "C" __global__ void mpas_cmbgb(int band,int points,int first,int weighted,const float* raw,const float* rwgt,float* out){
 // MPAS: module_ra_rrtmg_lw.F:8331-9982 (all 16 cmbgb routines).
 // Each thread owns one independent point and reduction. The original
 // order within sumk/sumf is preserved; interleaved independent sums
 // need no communication.
 int idx=blockIdx.x*blockDim.x+threadIdx.x;int groups=mpas_static_ngc[band];
 if(idx>=points*groups)return;
 int g=idx/points,point=idx%points,start=0,offset=0;
 for(int b=0;b<band;++b)offset+=mpas_static_ngc[b];
 for(int j=0;j<g;++j)start+=mpas_static_ngn[offset+j];
 float sum=0.0f;
 for(int j=0;j<mpas_static_ngn[offset+g];++j){int old=start+j;
   float term=raw[first?old+16*point:point+points*old];
   if(weighted)term=RLW_MU(term,rwgt[band*16+old]);
   sum=RLW_AD(sum,term);
 }
 out[first?g+groups*point:point+points*g]=sum;
}

"""

def initialize_device(raw):
    """MPAS: mpas_atmphys_rrtmg_lwinit.F:44-1593; module_ra_rrtmg_lw.F:7987-10623.

    Rust decodes pinned RRTMG_LW_DATA. raw maps rrlw_kgNN to its named
    arrays, each contiguous device float32 flattened in native Fortran
    order. This initializer performs all 16 g-point reductions and builds
    tau/exp/transition tables on CUDA. Immutable DATA tables are compiled
    as device constants; cp=1004.5 is the pinned MPAS framework value.
    Returned K owns the tables and stream readiness event; keep it in the
    driver's pools rather than hidden restart state.
    """
    import cupy as cp
    # Force module compilation and retain owners for constant global views.
    _gpu_kernel('mpas_init_tables')
    module=_MODULES[(int(cp.cuda.Device().id), MAX_RADIATION_LAYERS)]
    K={'band':{},'module':module}
    for name,size,integer in _STATIC_ARRAYS:
        address=module.get_global_var('mpas_static_'+name)
        if isinstance(address,tuple):
            address=address[0]
        memory=cp.cuda.UnownedMemory(address,size*4,module)
        K[name]=cp.ndarray((size,),dtype=cp.int32 if integer else cp.float32,
                           memptr=cp.cuda.MemoryPointer(memory,0))
    K.update({name:np.float32(value) for name,value in _STATIC_SCALARS.items()})
    K['rwgt']=cp.empty(256,dtype=cp.float32)
    _gpu_kernel('mpas_init_rwgt')((1,),(1,),(K['rwgt'],))
    for name in ('tau_tbl','exp_tbl','tfn_tbl'):
        K[name]=cp.empty(10001,dtype=cp.float32)
    _gpu_kernel('mpas_init_tables')((79,),(128,),
        (K['tau_tbl'],K['exp_tbl'],K['tfn_tbl']))
    reduced={}
    for band,dest,source,points,first,weighted in _REDUCTIONS:
        original=raw[f'rrlw_kg{band:02d}'][source]
        if (not isinstance(original,cp.ndarray) or original.dtype!=cp.float32
                or original.shape!=(points*16,) or not original.flags.c_contiguous):
            raise ValueError(f'raw {band}/{source} must preserve flattened Fortran words')
        groups=int(_legacy.NGC[band-1])
        out=cp.empty(points*groups,dtype=cp.float32)
        _gpu_kernel('mpas_cmbgb')(((out.size+127)//128,),(128,),
            (np.int32(band-1),np.int32(points),np.int32(first),np.int32(weighted),
             original,K['rwgt'],out))
        reduced[dest]=out
        if dest.endswith('/ka') or dest.endswith('/kb'):
            reduced[dest.rsplit('/',1)[0]+'/'+('absa' if dest.endswith('/ka') else 'absb')]=out
    for band,names in _legacy.GPU_BAND_TABS.items():
        arrays=[reduced[f'kg{band:02d}/{name}'] for name in names]
        # Pointer addresses are launch metadata, not decoded weather fields.
        pointers=cp.asarray(np.array([array.data.ptr for array in arrays],dtype=np.uint64))
        K['band'][band]=(arrays,pointers)
    K['bandptrs']=cp.asarray(np.array([K['band'][band][1].data.ptr for band in range(1,17)],dtype=np.uint64))
    K['reduced']=reduced
    K['ready']=cp.cuda.Event(disable_timing=True);K['ready'].record()
    return K
