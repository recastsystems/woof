"""Generate WOOF-ArWen's GFL (WRF v4.8.0, GFL 873b414) CUDA transcription.

Development tool only; production never parses or compiles Fortran.

Inputs (argv):
  1. tree directory: the GNU 15.2 ``-fdump-tree-original`` trees of the WRF
     v4.8.0 referee's own preprocessed compilation units
     (``phys/module_cu_gfl_{deep,sh,wrf}.f90`` from the strict build), compiled
     with the referee's flags plus ``-fdump-tree-original``;
  2. source directory: WRF v4.8.0 ``phys/GFL`` (the citation authority);
  3. output: the generated Python module (woof/core/gfl_873b414.py).

The lowering is the woof hex GFL transcription tool (tests/rrfs_gfl in
woof hex, used for its bf6a4b0 port), with WRF citations. The deep and
shallow modules are byte-identical between hex's bf6a4b0 and WRF's 873b414;
only the WRF driver file is WRF's own.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

from lower_gnu_tree import lower

# (preprocessed referee unit, original source path inside phys/GFL, citation path)
MODULES = (
    ('module_cu_gfl_deep.f90', 'module_cu_gfl_deep.F90', 'module_cu_gfl_deep.F90'),
    ('module_cu_gfl_sh.f90', 'module_cu_gfl_sh.F90', 'module_cu_gfl_sh.F90'),
    ('module_cu_gfl_wrf.f90', 'WRF/module_cu_gfl_wrf.F90', 'WRF/module_cu_gfl_wrf.F90'),
)
CITE = 'WRF v4.8.0 06d4240 phys/GFL 873b414'

DIMENSIONS = ('ims', 'ime', 'jms', 'jme', 'kms', 'kme', 'ids', 'ide', 'jds', 'jde', 'kds', 'kde',
              'its', 'ite', 'jts', 'jte', 'kts', 'kte')
INT_NAMES = {'itimestep', 'ichoice_deep', 'ishallow_g3', 'kpbl', 'k22_shallow', 'kbcon_shallow',
             'ktop_shallow'}
SCALARS = {'itimestep', 'dt', 'xlv', 'cp', 'g', 'r_v', 'ichoice_deep', 'ishallow_g3'}
# dxcell is a scalar in WRF's interface; the launcher passes it per column (stride 0
# for WRF's one dx, a column stride for a per-column dx). One column per thread means
# the column's own dx is the routine's dxCell.
SURFACES = {'dxcell', 'ht', 'xland', 'hfx', 'qfx', 'gsw', 'kpbl', 'htop', 'hbot', 'k22_shallow',
            'kbcon_shallow', 'ktop_shallow', 'xmb_shallow', 'raincv', 'pratec'}
OUTPUTS = ('htop', 'hbot', 'k22_shallow', 'kbcon_shallow', 'ktop_shallow', 'xmb_shallow',
           'raincv', 'pratec', 'gdc', 'gdc2', 'rthcuten', 'rqvcuten', 'rqccuten', 'rqicuten',
           'rucuten', 'rvcuten')


def arguments(text):
    match = re.search(r'subroutine cu_grell_freitas_li\((.*?)\)', text, re.I | re.S)
    return tuple(re.sub(r'[&\s]', '', match[1]).lower().split(','))


def generate(trees, sources, output):
    call_names = []
    for unit, _, _ in MODULES:
        call_names.extend(re.findall(r'^(?:void|real\(kind=4\)|integer\(kind=4\)) (\w+) \(',
                                     next(trees.glob(unit + '.*t.original')).read_text(), re.M))
    pieces, pins = [], []
    for unit, source_name, cite in MODULES:
        tree = next(trees.glob(unit + '.*t.original'))
        source = sources / source_name
        lowered = lower(tree.read_text(), source.read_text(), Path(source_name).name, call_names)
        lowered = lowered.replace(
            '// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/'
            + Path(source_name).name, '// ' + CITE + '/' + cite)
        if 'RRFS' in lowered or 'MPAS-Model' in lowered:
            raise ValueError(f'{unit}: an MPAS citation survived the WRF lowering')
        pieces.append(lowered)
        pins.append({'source': 'phys/GFL/' + source_name,
                     'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'referee_unit': 'phys/' + unit,
                     'referee_unit_sha256': hashlib.sha256((trees / unit).read_bytes()).hexdigest(),
                     'gnu_tree_sha256': hashlib.sha256(tree.read_bytes()).hexdigest(),
                     'lowered_sha256': hashlib.sha256(lowered.encode()).hexdigest()})
    args = arguments((trees / 'module_cu_gfl_wrf.f90').read_text())
    if any(name in args for name in ('num_chem', 'chem3d_in', 'pbl_scheme', 'areacell')):
        raise ValueError('the referee unit is not the WRF_CHEM=0 WRF driver')
    names = tuple(n for n in args if n not in DIMENSIONS)
    parts = []
    for name in args:
        if name in DIMENSIONS:
            parts.append('&' + name)
        else:
            typ = 'int' if name in INT_NAMES else 'GFReal'
            ix = names.index(name)
            parts.append(f'({typ}*)(ptrs[{ix}]+column*strides[{ix}])')
    dims = ('int ims=1,ime=1,jms=1,jme=1,kms=vertical[0],kme=vertical[1],ids=1,ide=2,jds=1,jde=2,'
            'kds=vertical[2],kde=vertical[3],its=1,ite=1,jts=1,jte=1,kts=vertical[4],kte=vertical[5];')
    entry = '''// ''' + CITE + '''/WRF/module_cu_gfl_wrf.F90:32-763 (one column per thread, WRF_CHEM=0)
__device__ void gfl_column(int column,const unsigned long long* ptrs,const unsigned long long* strides,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,const int* vertical,unsigned int* saved_fct_error){
#ifdef GF_HOST_WITNESS
 bool initial_error=saved_fct_error[0]!=0;
#else
 bool initial_error=atomicAdd(saved_fct_error,0u)!=0;
#endif
 GFWorkspace workspace{scratch,capacity,0,0,0,GFReal(0.0f),initial_error,nullptr,
  32768ULL*(unsigned long long)gf_max(vertical[1]-vertical[0]+1,vertical[5]-vertical[4]+1)+131072ULL};
 ''' + dims + '\n cu_grell_freitas_li(workspace, ' + ', '.join(parts) + ''');
 diagnostics[column]=workspace.diagnostics;highwater[column]=workspace.highwater;
#ifndef GF_HOST_WITNESS
 if(workspace.spill!=nullptr)free(workspace.spill);
#endif
 // ''' + CITE + '''/module_cu_gfl_deep.F90:2641-2705 (fct1d3's SAVE error flag)
 if(workspace.fct1d3_error){
#ifdef GF_HOST_WITNESS
 saved_fct_error[0]=1;
#else
 atomicOr(saved_fct_error,1u);
#endif
 }
}
#ifndef GF_HOST_WITNESS
extern "C" __global__ void gfl_columns(int ncol,const unsigned long long* ptrs,const unsigned long long* strides,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,const int* vertical,unsigned int* saved_fct_error){
 int column=blockIdx.x*blockDim.x+threadIdx.x;
 if(column<ncol)gfl_column(column,ptrs,strides,scratch+column*capacity,capacity,diagnostics,highwater,vertical,saved_fct_error);
}
#else
extern "C" void gfl_columns_host(int ncol,const unsigned long long* ptrs,const unsigned long long* strides,
 char* scratch,unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,const int* vertical,unsigned int* saved_fct_error){
 for(int c=0;c<ncol;c++)gfl_column(c,ptrs,strides,scratch,capacity,diagnostics,highwater,vertical,saved_fct_error);
}
#endif
'''
    support = Path(__file__).with_name('gfl_support.cuh').read_text()
    header = '''"""WRF v4.8.0 Grell-Freitas-Li (GFL 873b414), cu_physics=3: generated CUDA transcription.

GENERATED by tools/gfl_873b414/generate_wrf.py from the GNU original trees of
the WRF v4.8.0 strict referee's own compilation units. Do not edit by hand.
Every procedure body of module_cu_gfl_deep, module_cu_gfl_sh and the WRF
driver module_cu_gfl (WRF_CHEM=0) is retained.

Licence: GFL is NOAA code (US government); WRF's public-domain statement.
"""

from functools import lru_cache
from pathlib import Path

SCHEME = "cu_grell_freitas_li"
WRF_PIN = "06d4240ae989"
GFL_PIN = "873b414670323ae766ec4fe1f4b6b5ab891358ed"
COMPILER_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
'''
    header += ('ARGUMENTS = ' + repr(args) + '\nFIELDS = ' + repr(names)
               + '\nINTEGER_FIELDS = ' + repr(tuple(sorted(INT_NAMES)))
               + '\nSCALAR_FIELDS = ' + repr(tuple(sorted(SCALARS)))
               + '\nSURFACE_FIELDS = ' + repr(tuple(sorted(SURFACES)))
               + '\nOUTPUT_FIELDS = ' + repr(OUTPUTS) + '\n')
    header += 'SOURCE_HASHES = ' + repr({p['source']: p['source_sha256'] for p in pins}) + '\n'
    header += 'REFEREE_UNIT_HASHES = ' + repr({p['referee_unit']: p['referee_unit_sha256'] for p in pins}) + '\n\n'
    for name, text in (('_SUPPORT', support), ('_SCHEME_SOURCE', '\n'.join(pieces)), ('_ENTRY', entry)):
        header += name + " = r'''\n" + text + "\n'''\n\n"
    header += Path(__file__).with_name('launcher_wrf.py.in').read_text()
    output.write_text(header)
    out = Path('native')
    out.mkdir(exist_ok=True)
    (out / 'gfl_lowered.cuh').write_text('\n'.join(pieces))
    (out / 'gfl_entry.cuh').write_text(entry)
    (out / 'host.cpp').write_text('#define GF_HOST_WITNESS\n#include "gfl_support.cuh"\n'
                                  '#include "gfl_lowered.cuh"\n#include "gfl_entry.cuh"\n')
    scheme = '\n' + '\n'.join(pieces) + '\n'  # the generated module's _SCHEME_SOURCE value
    (out / 'transcription.json').write_text(json.dumps({
        'wrf': WRF_PIN_FULL, 'gfl': GFL_PIN_FULL,
        'scheme_source_sha256': hashlib.sha256(scheme.encode()).hexdigest(),
        'modules': pins}, indent=2) + '\n')
    print(f'Generated {len(names)} fields, {len(pieces)} complete modules')


WRF_PIN_FULL = '06d4240ae989'
GFL_PIN_FULL = '873b414670323ae766ec4fe1f4b6b5ab891358ed'

if __name__ == '__main__':
    generate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
