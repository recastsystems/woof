"""Use the native rank-one type for the plume module's 61 vectors.

These vectors otherwise carry a rank-four descriptor through every plume
substep. Bounds, stride, sequence association and overlap semantics stay intact.
"""
import re

VECTOR_SOURCE = r'''
template<class T> struct FVector {
 T *p=nullptr;static constexpr int rank=1;
 int lo[1]={1},hi[1]={0},stride[1]={1};
 SDDEV FVector()=default;
 SDDEV FVector(FArray<T> a){*this=a;}
 SDDEV void bind(int lower,int upper){lo[0]=lower;hi[0]=upper;}
 SDDEV FVector &operator=(FArray<T> a){
 if(a.rank!=1)sd_capacity_failure();
 p=a.p;lo[0]=a.lo[0];hi[0]=a.hi[0];stride[0]=a.stride[0];return *this;}
 SDDEV int size()const{return hi[0]-lo[0]+1;}
 SDDEV T &linear(int n)const{return p[n*stride[0]];}
 SDDEV T &operator()(int n)const{return p[(n-lo[0])*stride[0]];}
 SDDEV FArray<T> array()const{
 FArray<T> a;a.p=p;a.rank=1;a.lo[0]=lo[0];a.hi[0]=hi[0];a.stride[0]=stride[0];return a;}
 SDDEV operator FArray<T>()const{return array();}
 SDDEV FArray<T> section(Slice s)const{return array().section(s);}
};
template<class T> SDDEV FVector<T> sd_sequence(FVector<T> a,int i){
 FVector<T> r;r.p=&a(i);r.bind(1,1);return r;}
template<class T> SDDEV FArray<T> sd_expr(FVector<T> a){return a.array();}
template<class T,class E> SDDEV void sd_assign(FVector<T> a,E b,Arena &arena){
 if constexpr(SdScalar<E>::value){for(int j=0;j<a.size();j++)a.linear(j)=b;}
 else sd_assign(a.array(),b,arena);
}
template<class T> SDDEV void sd_assign(FVector<T> a,FVector<T> b,Arena &arena){
 // Native array assignment needs staging only when the storage overlaps.
 auto ab=(unsigned long long)a.p,bb=(unsigned long long)b.p;
 auto ae=ab+(a.size()?((a.size()-1)*a.stride[0]+1)*sizeof(T):0);
 auto be=bb+(b.size()?((b.size()-1)*b.stride[0]+1)*sizeof(T):0);
 if(ae<=bb || be<=ab){for(int j=0;j<a.size();j++)a.linear(j)=b.linear(j);}
 else sd_assign(a.array(),b.array(),arena);
}
'''


# The native derived type declares these as fixed-size arrays, not pointers
# or assumed-shape dummy arguments (module_zero_plumegen_coms.F90:9-55).
FIXED_VECTOR_SOURCE = r'''
template<class T> struct FPlumeVector {
 mutable T p[200];
 static constexpr int rank=1,lo[1]={1},hi[1]={200},stride[1]={1};
 SDDEV FPlumeVector &operator=(FArray<T> a){
 if(a.rank!=1 || a.lo[0]!=1 || a.hi[0]!=200 || a.stride[0]!=1)sd_capacity_failure();
 for(int j=0;j<200;j++)p[j]=a.linear(j);return *this;}
 SDDEV int size()const{return 200;}
 SDDEV T &linear(int n)const{return p[n];}
 SDDEV T &operator()(int n)const{return p[n-1];}
 SDDEV FVector<T> vector()const{FVector<T> a;a.p=p;a.bind(1,200);return a;}
 SDDEV FArray<T> array()const{return vector().array();}
 SDDEV operator FVector<T>()const{return vector();}
 SDDEV operator FArray<T>()const{return array();}
 SDDEV FArray<T> section(Slice s)const{return array().section(s);}
};
template<class T> SDDEV FVector<T> sd_sequence(const FPlumeVector<T> &a,int i){return sd_sequence(a.vector(),i);}
template<class T> SDDEV FArray<T> sd_expr(const FPlumeVector<T> &a){return a.array();}
template<class T,class E> SDDEV void sd_assign(const FPlumeVector<T> &a,E b,Arena &arena){sd_assign(a.vector(),b,arena);}
template<class T> SDDEV void sd_assign(const FPlumeVector<T> &a,const FPlumeVector<T> &b,Arena &arena){sd_assign(a.vector(),b.vector(),arena);}
'''


def source(text, routines):
    begin = text.index('struct Plume {')
    end = text.index('\n};', begin)
    block = text[begin:end]
    vectors = re.findall(r'FArray<float> (\w+);', block)
    if len(vectors) != 61:
        raise ValueError('Pinned plume vector inventory changed: ' + str(len(vectors)))
    for name in vectors:
        # Every original module allocation is rank one. Refuse a graph change
        # that would make the specialized descriptor lose a dimension.
        allocation = 'plume.' + name + ' = storage.alloc<float>(1,200);'
        if text.count(allocation) != 1:
            raise ValueError('Pinned plume vector allocation changed: ' + name)
        # Native fixed arrays reside inside their derived type. Their original
        # module initialization fills them after pointer association is made.
        text = text.replace(allocation, '// Fixed native Plume field: ' + name, 1)
    block = block.replace('FArray<float>', 'FPlumeVector<float>')
    text = text[:begin] + VECTOR_SOURCE + FIXED_VECTOR_SOURCE + block + text[end:]
    # Retain declared native rank at routine boundaries and SAVE storage.
    # Rank-three plume inputs keep their full descriptor.
    start = re.search(r'^SDDEV void plumerise\([^\n]+\) \{', text, re.M).start()
    stop = text.index('SDDEV void aero_rad_init(State &state,Arena &arena) {')
    names = re.findall(r'^SDDEV (?:void|float) (\w+)\([^\n]+\) \{', text[start:stop], re.M)
    for name in names:
        row = routines[name]
        arguments = [arg for arg in row['args']
                     if len(row['symbols'][arg]['dim']) == 1 and row['symbols'][arg]['type'] == 'float']
        if not arguments:
            continue
        pattern = r'(SDDEV (?:void|float) ' + re.escape(name) + r'\()([^\n]+)(\)[; {])'
        def signature(match):
            args = match.group(2)
            for arg in arguments:
                old = 'FArray<float> ' + arg
                args, count = re.subn(re.escape(old) + r'(?=,|$)', 'FVector<float> ' + arg, args)
                if count != 1:
                    raise ValueError('Pinned native plume argument changed: ' + name + '.' + arg)
            return match.group(1) + args + match.group(3)
        text, count = re.subn(pattern, signature, text)
        if count != 2:
            raise ValueError('Pinned native plume declaration/definition changed: ' + name)
    return text
