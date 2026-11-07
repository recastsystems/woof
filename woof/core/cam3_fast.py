"""Fast CAM internal REAL32 image; strict CAM keeps the native REAL(R8) image.

Only production tiers (explicit longwave and shortwave selectors) use this
transform. The native argument ABI and saved-field byte offsets stay fixed.
The Rust reader packs REAL(R8) fields as REAL32 at those offsets. Workspace
bounds remain the conservative native bounds; the absorption arena uses the
active scalar width. Standalone oracle entry points keep their native image.
"""
import re


def _implicit_aerosol(source):
    # Native aer_trn assigns the entire seven-band tensor to one when
    # strat_volcanic is false. Supply that exact value at each read instead
    # of writing and reading seven quadratic arrays per column. Keep the
    # volcanic calculation and its tensor when that switch is enabled.
    marker='__device__ int aer_trn_(CamState* state, doublereal *aer_mpp__'
    cut=source.rindex(marker)
    prefix,count=re.subn(r'(?<!&)aer_trn_ttl__\[([^\]]+)\]',
                        r'(state->cg_strat_volcanic__?aer_trn_ttl__[\1]:1.f)',source[:cut])
    if count!=29:raise ValueError('CAM aerosol read contract changed')
    source=prefix+source[cut:]
    call='    aer_trn_(state, aer_mpp__, aer_trn_ttl__, pcols, pver, pverp);'
    if source.count(call)!=1:raise ValueError('CAM aerosol call contract changed')
    return source.replace(call,'    if (state->cg_strat_volcanic__) '+call.strip())


def fast_source(source):
    source=re.sub(r'\bdouble\b','float',source)
    source=source.replace('(8/sizeof(T))','(4/sizeof(T))')
    # Gases are the sole binary64 driver operand; the external ABI stays fixed.
    source=re.sub(r'float\* gases(?=[,)])','double* gases',source)
    # Native uses this threshold only to distinguish zero cloud fraction.
    # Its positive REAL32 equivalent must not round to zero.
    source=re.sub(r'\b1e-80\b','0x1p-149f',source)
    tokens=re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\b0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?\d+[fFlL]?|(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)(?:[eE][+-]?\d+)?[fFlL]?')
    def literal(m):
        t=m.group()
        if t.startswith(('/', '"', '0x', '0X')) or t[-1:] in 'fFlL':return t
        return t+'f'
    source=tokens.sub(literal,source)
    for a,b in [('exp','expf'),('log','logf'),('pow','powf'),('sqrt','sqrtf'),('log10','log10f'),('floor','floorf'),('trunc','truncf'),('frexp','frexpf'),('round','roundf'),('cos','cosf'),('sin','sinf')]:
        source=re.sub(r'\b'+a+r'\b',b,source)
    # Native routines fill private scratch before using its defined words.
    # Keep flux-accumulator clears; avoid blanket strided stores to every
    # private array on each nested call. Strict retains its original clears.
    source=source.replace('for(unsigned long long i=0;i<n;++i)out[i]=T{};','')
    source=_implicit_aerosol(source)
    # Absorption workers only read a small part of their parent's header.
    # A private header lets the compiler discard unused members instead of
    # writing the entire header to global scratch for every level worker.
    header='CamState* state=(CamState*)(workspace+thread*CAM_ABS_WORK_BYTES);\n*state=*(const CamState*)(parents+column*CAM_WORK_BYTES);'
    if source.count(header)!=1:raise ValueError('CAM absorption header contract changed')
    source=source.replace(header,'CamState private_state=*(const CamState*)(parents+column*CAM_WORK_BYTES);\nCamState* state=&private_state;')
    # Standalone oracle entry points need not inflate the fast runtime image.
    for name in ('cam_emissivity','cam_lw','cam_sw','cam_log10_probe','cam_log10_variants'):
        match=re.search(r'extern "C" __global__[^\n]*? void '+name+r'\(',source)
        if not match:continue
        start=match.start();at=source.index('{',match.end())+1;depth=1
        while depth:
            depth+=(source[at]=='{')-(source[at]=='}');at+=1
        source=source[:start]+source[at:]
    return source

