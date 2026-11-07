"""Interleave private WRF arrays across independent CUDA columns."""
import re

HEADER=r'''
template<class T> struct M471Ptr;
template<class T> struct M471Ref {
 T *p; int stride;
 __device__ operator T() const{return *p;}
 __device__ T operator=(T x){*p=x;return x;}
 __device__ T operator=(const M471Ref&x){return *this=(T)x;}
 template<class U> __device__ T operator+=(U x){return *this=*p+x;}
 template<class U> __device__ T operator-=(U x){return *this=*p-x;}
 template<class U> __device__ T operator*=(U x){return *this=*p*x;}
 template<class U> __device__ T operator/=(U x){return *this=*p/x;}
 __device__ M471Ptr<T> operator&() const;
};
template<class T> struct M471Ptr {
 T *p; int stride;
 __device__ M471Ptr(T*q=nullptr,int s=1):p(q),stride(s){}
 __device__ T& operator*()const{return *p;}
 __device__ M471Ref<T> operator[](int i)const{return {p+i*stride,stride};}
 __device__ M471Ptr operator+(int i)const{return {p+i*stride,stride};}
 __device__ M471Ptr operator-(int i)const{return {p-i*stride,stride};}
 __device__ M471Ptr& operator+=(int i){p+=i*stride;return *this;}
 __device__ M471Ptr& operator-=(int i){p-=i*stride;return *this;}
 __device__ M471Ptr& operator--(){p-=stride;return *this;}
 __device__ M471Ptr& operator++(){p+=stride;return *this;}
};
template<class T> __device__ M471Ptr<T> M471Ref<T>::operator&()const{return {p,stride};}
template<class T> __device__ M471Ptr<T> m471_array(unsigned char*base,int offset,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 return {(T*)(base+(unsigned long long)offset*n)+c,n};
}
'''


def direct_arguments(engine):
    """Bind native array arguments directly to distinct packed slab ranges."""
    loads=re.compile(r'#pragma unroll 1\nfor \(int k=0;k<[^;]+;\+\+k\)(a_\w+)\[k\]=(f32|i32)\[\((.+)\+k\)\*n\+c\];')
    bindings={}
    def bind(m):
        bindings[m[1]]=(m[2],m[3])
        return ''
    engine=loads.sub(bind,engine)
    for var,(slab,offset) in bindings.items():
        pattern=r'M471Ptr<(float|int)> '+var+r'=m471_array<\1>\(m471_work,[^;]+;'
        engine,count=re.subn(pattern,lambda m:f'M471Ptr<{m[1]}> {var}={{{slab}+{offset}*n+c,n}};',engine)
        if count!=1:raise ValueError('argument declaration missing: '+var)
        stores=r'#pragma unroll 1\nfor \(int k=0;k<[^;]+;\+\+k\)'+slab+r'\[\(.+\+k\)\*n\+c\]='+var+r'\[k\];'
        engine=re.sub(stores,'',engine)
    if len(bindings)<60:raise ValueError('argument array bindings incomplete')
    return engine

def coalesce(text):
    # Every numeric pointer in the generated routines is either a scalar
    # (stride one) or a private array (stride n). Pointer offset arithmetic,
    # array sections and references retain that stride through calls.
    start=text.index('typedef int integer;')
    end=text.index('extern "C" __global__')
    prefix,body,engine=text[:start],text[start:end],text[end:]
    body=re.sub(r'\b(real|integer|logical|doublereal|float|int|double)\s*\*\s*(\w+)',r'M471Ptr<\1> \2',body)
    # Scratch declarations have a raw byte offset. Their new physical
    # address is offset*n plus the lane's typed word, never offset+c*size.
    pat=r'M471Ptr<(\w+)> (\w+)=\(\1\*\)\(m471_work\+([^;]+)\);'
    body=re.sub(pat,r'M471Ptr<\1> \2=m471_array<\1>(m471_work,\3,m471_n);',body)
    body=body.replace('unsigned char *m471_work,','unsigned char *m471_work, int m471_n,')
    body=body.replace('(m471_work,','(m471_work,m471_n,')
    # Array constructors already take their width as the final argument.
    body=re.sub(r'm471_array<(\w+)>\(m471_work,m471_n,',r'm471_array<\1>(m471_work,',body)
    engine=re.sub(r'unsigned char\*m471_work=arena\+c\*[^;]+;', 'unsigned char*m471_work=arena;int m471_n=n;',engine)
    engine=re.sub(r'\b(float|int) \*(\w+)=\(\1\*\)\(m471_work\+([^;]+)\);',r'M471Ptr<\1> \2=m471_array<\1>(m471_work,\3,n);',engine)
    engine=engine.replace('mynnedmf_(m471_work,','mynnedmf_(m471_work,n,')
    engine=direct_arguments(engine)
    marker='typedef double doublereal;'
    body=body.replace(marker,marker+HEADER,1)
    from phases import phases
    from plume_parallel import plume_parallel
    from fold_locals import fold_locals
    return fold_locals(plume_parallel(phases(prefix+body+engine)))

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(coalesce(a.input.read_text()))
