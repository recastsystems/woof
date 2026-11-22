"""Lane-interleave the GF column arrays, the way CUDA lays out local memory.

The transcribed scheme gives every thread one contiguous scratch slab, so a
warp reading ``a[k]`` touches 32 different cache lines (32-way scatter).
WOOF-ArWen's GF measured the same placement at 2-6x slower than an interleaved
one (woof/core/kernels/gf.cu, GFWS_LANES). Here every float/integer array
pointer in the scheme becomes a lane pointer whose element ``i`` lives at
``p + i*GF_LANES``: element ``e`` of lane ``t`` in a 128-column slab sits at
``slab + (e*GF_LANES + t)*4``, so a warp reading ``a[k]`` reads 32
consecutive words. Addressing stays linear in the element index, so
``&a[e]`` handed to a callee and indexed there by ``j`` reaches element
``e+j`` exactly as before; the arithmetic, its order and every index
expression (including native's own out-of-range reads) are unchanged. Only
placement moves. Character arrays (``ierrc``) keep a small contiguous
per-thread region. The host witness compiles with ``GF_LANES 1`` and is the
original contiguous layout.

This transform runs once at kernel setup; nothing here runs in a step.
"""
import re

LANES = 128

#: Resident 128-thread blocks per SM the column kernel is compiled for.
#: Measured on the 60->15 km zoom capture (216,187 columns, step 19), quiet
#: RTX PRO 6000, 2026-10-24: fast 2 blocks (255 regs) 26.6 ms, 3 (168) 27.9,
#: 4 (128) 29.8, 5 30.8, 6 31.4; strict 4 blocks (128 regs) 38.7 ms. Fewer
#: blocks also means fewer workspace slots (less scratch).
_DEFINES = r'''#ifndef GF_MIN_BLOCKS
#ifdef HEX_NUMERICS_FAST
#define GF_MIN_BLOCKS 2
#else
#define GF_MIN_BLOCKS 4
#endif
#endif
'''

#: Separates the scheme from the column entry (host tests swap index stubs in).
ENTRY_MARKER = "// GF column entry"

_LANE_POINTER = r'''
#ifndef GF_LANES
#ifdef GF_HOST_WITNESS
#define GF_LANES 1
#else
#define GF_LANES 128
#endif
#endif
#ifdef GF_HOST_WITNESS
#define __forceinline__ inline
#define __launch_bounds__(...)
#endif
// A column array pointer whose element i lives GF_LANES words after element
// i-1 (lane-interleaved placement). Arithmetic and dereference stay linear in
// the element index; a pointer to a scalar is only ever read at element 0.
template<class T> struct GFLanePtr {
  T* p;
  __device__ __forceinline__ GFLanePtr() = default;
  __device__ __forceinline__ GFLanePtr(T* q):p(q){}
  __device__ __forceinline__ explicit GFLanePtr(void* q):p((T*)q){}
  __device__ __forceinline__ GFLanePtr(decltype(nullptr)):p(nullptr){}
  // Fortran OPTIONAL arguments are tested against null.
  __device__ __forceinline__ bool operator==(decltype(nullptr)) const {return p==nullptr;}
  __device__ __forceinline__ bool operator!=(decltype(nullptr)) const {return p!=nullptr;}
  template<class I> __device__ __forceinline__ T& operator[](I i) const {return p[(long long)i*GF_LANES];}
  __device__ __forceinline__ T& operator*() const {return *p;}
  template<class I> __device__ __forceinline__ GFLanePtr operator+(I i) const {return GFLanePtr(p+(long long)i*GF_LANES);}
};
typedef GFLanePtr<GFReal> GFP;
typedef GFLanePtr<int> GIP;
'''

_WORKSPACE_OLD = '''struct GFWorkspace {
  char* data;unsigned long long capacity,position,highwater;unsigned int diagnostics;
  __device__ void* allocate(unsigned long long n){
    n=(n+15)&~15ULL;
    if(position+n>capacity){gf_trap();return nullptr;}
    void* result=data+position;position+=n;highwater=gf_max(highwater,position);return result;
  }
};
struct GFFrame {
 GFWorkspace &ws;unsigned long long saved;
 __device__ GFFrame(GFWorkspace &w):ws(w),saved(w.position){}
 __device__ ~GFFrame(){ws.position=saved;}
};'''

# Positions count this lane's bytes exactly as before (so capacity and
# highwater keep their meaning); a lane byte offset b is the address
# data + b*GF_LANES, i.e. element b/4 of the lane-interleaved slab.
_WORKSPACE_NEW = '''struct GFWorkspace {
  char* data;unsigned long long capacity,position,highwater;unsigned int diagnostics;
  char* bytes;unsigned long long bcapacity,bposition;
  __device__ void* allocate(unsigned long long n){
    n=(n+15)&~15ULL;
    if(position+n>capacity){gf_trap();return nullptr;}
    void* result=data+position*GF_LANES;position+=n;highwater=gf_max(highwater,position);return result;
  }
  // Character arrays are read bytewise: contiguous per thread.
  __device__ void* allocate_bytes(unsigned long long n){
    n=(n+15)&~15ULL;
    if(bposition+n>bcapacity){gf_trap();return nullptr;}
    void* result=bytes+bposition;bposition+=n;return result;
  }
};
struct GFFrame {
 GFWorkspace &ws;unsigned long long saved,bsaved;
 __device__ GFFrame(GFWorkspace &w):ws(w),saved(w.position),bsaved(w.bposition){}
 __device__ ~GFFrame(){ws.position=saved;ws.bposition=bsaved;}
};'''

#: Contiguous character bytes per thread (ierrc and ierrcs: one 50-byte
#: element each at its=ite=1, 16-byte rounded).
CHARACTER_BYTES = 256


def lower(support, scheme, entry):
    """Return the lane-interleaved CUDA source (support + scheme + entry)."""
    assert support.count(_WORKSPACE_OLD) == 1, "GF workspace support changed"
    support = support.replace(_WORKSPACE_OLD, _WORKSPACE_NEW)
    support = support.replace("typedef char GFText[50];", "typedef char GFText[50];\n" + _LANE_POINTER, 1)
    assert "GFLanePtr" in support
    # Character arrays: their descriptors take the contiguous byte region.
    text = set(re.findall(r'\b\w+ = \(GFText \*\) (D_\d+);', scheme))
    assert len(text) == 2, text
    for name in text:
        pattern = name + ' = (void *) workspace.allocate ('
        assert scheme.count(pattern) == 1, name
        scheme = scheme.replace(pattern, name + ' = (void *) workspace.allocate_bytes (')
    # Every float/integer pointer in the scheme is a lane pointer. Fixed-size
    # local arrays (xff_shal, seed, ...) stay C arrays: none is passed on.
    before = len(re.findall(r'\bGFReal \*|\bint \*', scheme))
    scheme = re.sub(r'\bGFReal \*', 'GFP ', scheme)
    scheme = re.sub(r'\bint \*', 'GIP ', scheme)
    assert before > 600 and not re.search(r'\bGFReal \*|\bint \*', scheme)
    # Entry: level payloads are lane-interleaved [slab][field][level][lane].
    old_column = '__device__ void gf_column(int nz,GFReal* lev,GFReal* surface,int* isurface,\n GFReal* outlev,'
    assert entry.count(old_column) == 1
    entry = entry.replace(old_column, '__device__ void gf_column(int nz,GFP lev,GFReal* surface,int* isurface,\n GFP outlev,')
    kernel = entry[entry.index('#ifndef GF_HOST_WITNESS\nextern "C" __global__ void gf_mpas_columns'):entry.index('#else\nextern "C" void gf_port_host')]
    entry = entry.replace(kernel, '''#ifndef GF_HOST_WITNESS
// One thread per workspace slot, looping over positions slot, slot+T, ...
// (T = gridDim*blockDim, a multiple of GF_LANES, so a slot keeps its lane).
// Level payloads are stored by position; order (optional) maps a position to
// its column, so warps can hold columns of like activity. Columns are
// independent; each executes the complete source order.
extern "C" __global__ void __launch_bounds__(GF_LANES, GF_MIN_BLOCKS) gf_mpas_columns(int ncols,int nz,float* lev,float* surface,int* isurface,
 float* outlev,float* outsurface,int* outint,const int* order,
 int c0,int c1,int c2,int c3,float k0,float k1,float k2,float k3,
 unsigned char* scratch,unsigned long long scratch_bytes,unsigned int* diagnostics,
 unsigned long long* highwater){
 // Driver words arrive by value (no host-to-device copy, so no stream sync
 // per call); the scheme reads them through pointers, at element 0 only.
 int control[4]={c0,c1,c2,c3};
 float constants[4]={k0,k1,k2,k3};
 const long long slot=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const long long slots=(long long)gridDim.x*blockDim.x;
 const int lane=(int)(slot%GF_LANES);
 char* slab=(char*)scratch+(slot/GF_LANES)*(scratch_bytes+GF_CHARACTER_BYTES)*GF_LANES;
 for(long long col=slot;col<ncols;col+=slots){
   const long long colslab=col/GF_LANES;
   GFWorkspace ws={slab+4*lane,scratch_bytes,0,0,0,
     slab+scratch_bytes*GF_LANES+(long long)lane*GF_CHARACTER_BYTES,GF_CHARACTER_BYTES,0};
   const long long c=order?(long long)order[col]:col;
   gf_column(nz,GFP((GFReal*)lev+colslab*15*nz*GF_LANES+lane),(GFReal*)surface+c*8,isurface+c,
   GFP((GFReal*)outlev+colslab*6*nz*GF_LANES+lane),(GFReal*)outsurface+c*6,outint+c*3,
   control,(GFReal*)constants,ws);
   diagnostics[c]=ws.diagnostics;highwater[c]=ws.highwater;
 }
}
''')
    host_ws = ' GFWorkspace ws={scratch,capacity,0,0,0};'
    assert entry.count(host_ws) == 1
    entry = entry.replace(host_ws, ' char characters[GF_CHARACTER_BYTES];\n GFWorkspace ws={scratch,capacity,0,0,0,characters,GF_CHARACTER_BYTES,0};')
    entry = entry.replace('gf_column(nz,(GFReal*)lev,(GFReal*)surface,isurface,(GFReal*)outlev,',
                          'gf_column(nz,GFP((GFReal*)lev),(GFReal*)surface,isurface,GFP((GFReal*)outlev),')
    assert 'GFP((GFReal*)lev)' in entry
    defines = f"#define GF_CHARACTER_BYTES {CHARACTER_BYTES}\n" + _DEFINES
    return defines + support + "\n" + scheme + "\n" + ENTRY_MARKER + "\n" + entry
