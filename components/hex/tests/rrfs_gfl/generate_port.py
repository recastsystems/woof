"""Generate every GFL procedure from strict GNU original trees and pin the result."""
from pathlib import Path
import hashlib
import json
import sys
import re
from lower_gnu_tree import lower
from abi import arguments, DIMENSIONS, INT_NAMES, SCALARS, SURFACES, OUTPUTS

MODULES = ('module_cu_gfl_deep.F90', 'module_cu_gfl_sh.F90', 'module_cu_gfl_mpas.F90')


def generate(root, output):
    pieces=[]
    pins=[]
    call_names=[]
    for name in MODULES:
        call_names.extend(re.findall(r'^(?:void|real\(kind=4\)|integer\(kind=4\)) (\w+) \(', next(root.glob(name+'.*t.original')).read_text(),re.M))
    for name in MODULES:
        tree=next(root.glob(name+'.*t.original'))
        source=Path('native')/name
        path=('MPAS/' if name.endswith('mpas.F90') else '')+name
        citation_source = source.read_text()
        if name.endswith('mpas.F90'):
            # Citations refer to the pinned fork, before the separately labelled
            # one-line correction. The GNU tree still includes that correction.
            correction = '       ierrm(i)    = 0\n'
            if citation_source.count(correction) != 1:
                raise ValueError('approved native wrapper correction missing')
            citation_source = citation_source.replace(correction, '')
        lowered=lower(tree.read_text(), citation_source, path, call_names)
        if name.endswith('mpas.F90'):
            original='dx_loc[(long long) *its + offset_399] < 4.5e+3f'
            assert lowered.count(original)==1
            lowered=lowered.replace(original,'workspace.row_dx_first < 4.5e+3f')
        pieces.append(lowered)
        pins.append({'source':path,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'gnu_tree_sha256':hashlib.sha256(tree.read_bytes()).hexdigest(),
                     'lowered_sha256':hashlib.sha256(lowered.encode()).hexdigest()})
    args=arguments('native/module_cu_gfl_mpas.F90')
    names=tuple(n for n in args if n not in DIMENSIONS and n != 'pbl_scheme')
    parts=[]
    for name in args:
        if name in DIMENSIONS:
            parts.append('&'+name)
        elif name == 'pbl_scheme':
            parts.append('(char*)pbl')
        else:
            typ='int' if name in INT_NAMES else 'GFReal'
            ix=names.index(name)
            parts.append(f'({typ}*)(ptrs[{ix}]+column*strides[{ix}])')
    dims='int ims=1,ime=1,jms=1,jme=1,kms=vertical[0],kme=vertical[1],ids=1,ide=2,jds=1,jde=2,kds=vertical[2],kde=vertical[3],its=1,ite=1,jts=1,jte=1,kts=vertical[4],kte=vertical[5];'
    entry='''// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_mpas.F90:33-745
__device__ void gfl_column(int column,int nz,const unsigned long long* ptrs,const unsigned long long* strides,const char* pbl,long long pbl_len,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,GFReal row_dx_first,const int* vertical,unsigned int* saved_fct_error){
#ifdef GF_HOST_WITNESS
 bool initial_error=saved_fct_error[0]!=0;
#else
 bool initial_error=atomicAdd(saved_fct_error,0u)!=0;
#endif
 GFWorkspace workspace{scratch,capacity,0,0,0,row_dx_first,initial_error,nullptr,
  32768ULL*(unsigned long long)gf_max(vertical[1]-vertical[0]+1,vertical[5]-vertical[4]+1)+131072ULL};
 '''+dims+'\n cu_grell_freitas_li(workspace, '+', '.join(parts)+''',pbl_len);
 diagnostics[column]=workspace.diagnostics;highwater[column]=workspace.highwater;
#ifndef GF_HOST_WITNESS
 if(workspace.spill!=nullptr)free(workspace.spill);
#endif
 // MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:2641-2705
 if(workspace.fct1d3_error){
#ifdef GF_HOST_WITNESS
 saved_fct_error[0]=1;
#else
 atomicOr(saved_fct_error,1u);
#endif
 }
}
#ifndef GF_HOST_WITNESS
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_mpas.F90:33-745
extern "C" __global__ void gfl_columns(int ncol,int nz,const unsigned long long* ptrs,const unsigned long long* strides,const char* pbl,long long pbl_len,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,const float* row_dx_first,const int* vertical,unsigned int* saved_fct_error){
 int column=blockIdx.x*blockDim.x+threadIdx.x;
 if(column<ncol)gfl_column(column,nz,ptrs,strides,pbl,pbl_len,scratch+column*capacity,capacity,diagnostics,highwater,GFReal(row_dx_first[0]),vertical,saved_fct_error);
}
#else
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_mpas.F90:33-745
extern "C" void gfl_columns_host(int ncol,int nz,const unsigned long long* ptrs,const unsigned long long* strides,const char* pbl,long long pbl_len,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,float row_dx_first,const int* vertical,unsigned int* saved_fct_error){
 for(int c=0;c<ncol;c++)gfl_column(c,nz,ptrs,strides,pbl,pbl_len,scratch,capacity,diagnostics,highwater,GFReal(row_dx_first),vertical,saved_fct_error);
}
#endif
'''
    header='''"""Pinned RRFS GFL bf6a4b0: complete CUDA transcription and MPAS boundary.

Authority RRFS-MPAS 1d6e370e + ierrm init (owner ruling 2026-10-14 07:45Z).
Separately labelled source-corrected strict GNU single precision. All 31
procedure bodies retained. See receipts for measured identity scope.
"""
from functools import lru_cache
from pathlib import Path

SCHEME = "cu_grell_freitas_li"
BACKEND = "hex-cuda"
ORACLE_SCHEME = "gfl_bf6a4b0"
SOURCE_PIN = "1d6e370e7b43a332074809f170532a216c2212fb"
GFL_PIN = "bf6a4b0d58f112322fbf9b11d43ed2027aeb56fa"
AUTHORITY_LABEL = "RRFS fork 1d6e370e + ierrm init (owner ruling 2026-10-14 07:45Z)"
AUTHORITY_PATCH_SHA256 = "8dcf2f4c31501f3a72a72f72f250383290bfa95df1130179c2986698ddaeb157"
COMPILER_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
'''
    header+='ARGUMENTS = '+repr(args)+'\nFIELDS = '+repr(names)+'\nINTEGER_FIELDS = '+repr(tuple(sorted(INT_NAMES)))+'\nSCALAR_FIELDS = '+repr(tuple(sorted(SCALARS)))+'\nSURFACE_FIELDS = '+repr(tuple(sorted(SURFACES)))+'\nOUTPUT_FIELDS = '+repr(OUTPUTS)+'\n'
    header+='SOURCE_HASHES = '+repr({p['source']:p['source_sha256'] for p in pins})+'\n'
    for name,text in (('_SUPPORT',Path(__file__).with_name('gfl_support.cuh').read_text()),('_SCHEME_SOURCE','\n'.join(pieces)),('_ENTRY',entry),('_DRIVER',Path(__file__).with_name('gfl_driver.cuh').read_text())):
        header+=name+" = r'''\n"+text+"\n'''\n\n"
    header+=Path(__file__).with_name('launcher.py').read_text()
    output.write_text(header)
    Path('native/gfl_lowered.cuh').write_text('\n'.join(pieces))
    Path('native/gfl_entry.cuh').write_text(entry)
    Path('native/host.cpp').write_text('#define GF_HOST_WITNESS\n#include "gfl_support.cuh"\n#include "gfl_lowered.cuh"\n#include "gfl_entry.cuh"\n')
    Path('receipts/transcription.json').write_text(json.dumps({'pin':'bf6a4b0d58f112322fbf9b11d43ed2027aeb56fa', 'modules':pins},indent=2)+'\n')
    print(f'Generated {len(names)} fields, {len(pieces)} complete modules')

if __name__=='__main__':
    generate(Path(sys.argv[1]),Path(sys.argv[2]))
