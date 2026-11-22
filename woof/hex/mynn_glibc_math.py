"""Assemble shared glibc device arithmetic for RKIND MYNN callers.

Existing WOOF-ArWen headers are read from the pinned engine, without modifying them.
New CORE-MATH erff/log10f text is shared here. No CUDA libm fallback is used.
"""
from importlib.resources import files
from pathlib import Path


def mynn_glibc_source():
    """Bind the float operations present in the pinned MYNN native object.

    The native strict object uses only binary32 libm calls. ARRAY_SUPPORT's
    binary64 declarations remain unresolved unless a caller supplies them;
    a newly introduced double path therefore cannot silently use CUDA math.
    """
    root=files('woof.core.kernels')
    parts=['namespace hex_mynn_glibc { typedef float real;\n',
           root.joinpath('mynn_libm.cuh').read_text(), '\n}\n',
           'namespace hex_mynn_trig {\n',
           root.joinpath('glibc_trig_flt32.cuh').read_text(), '\n}\n',
           root.joinpath('sfire_libm.cuh').read_text(),
           Path(__file__).with_name('glibc_erff.cuh').read_text(),
           Path(__file__).with_name('glibc_log10f.cuh').read_text()]
    for name,callee in {
        'exp':'hex_mynn_glibc::mynn_glibc_expf',
        'log':'hex_mynn_glibc::mynn_glibc_logf',
        'log10':'hex_glibc_log10f', 'atan':'hex_mynn_trig::glibc_atanf',
        'tanh':'sfire_tanhf', 'erf':'hex_glibc_erff',
    }.items():
        special = ''
        if name in ('exp','log','tanh'):
            special = 'unsigned int ux=__float_as_uint(x); if((ux&0x7fffffffu)>0x7f800000u) return __uint_as_float(ux|0x00400000u); '
        if name == 'log':
            special += 'if((ux&0x80000000u) && (ux&0x7fffffffu)!=0u) return __uint_as_float(0xffc00000u); '
        parts.append(f'\n__device__ float mynn_{name}(float x) {{{special}return {callee}(x);}}\n')
    parts.append(r'''
__device__ float mynn_pow(float x,float y) {
    unsigned int ux=__float_as_uint(x),uy=__float_as_uint(y);
    unsigned int ax=ux&0x7fffffffu,ay=uy&0x7fffffffu;
    if(ay==0u) return (ax>0x7f800000u && !(ux&0x00400000u)) ? __uint_as_float(ux|0x00400000u) : 1.0f;
    if(ux==0x3f800000u) return (ay>0x7f800000u && !(uy&0x00400000u)) ? __uint_as_float(uy|0x00400000u) : 1.0f;
    if(ax>0x7f800000u) {
        unsigned int result=ux|0x00400000u;
        if((ux&0x80000000u) && ay!=0u && ay<0x7f800000u && hex_mynn_glibc::mynn_glibc_powf_checkint(uy)==1) result^=0x80000000u;
        return __uint_as_float(result);
    }
    if(ay>0x7f800000u) return __uint_as_float(uy|0x00400000u);
    if((ux&0x80000000u) && ax!=0u && ax<0x7f800000u && ay<0x7f800000u && hex_mynn_glibc::mynn_glibc_powf_checkint(uy)==0)
        return __uint_as_float(0xffc00000u);
    return hex_mynn_glibc::mynn_glibc_powf(x,y);
}
''')
    return ''.join(parts)
