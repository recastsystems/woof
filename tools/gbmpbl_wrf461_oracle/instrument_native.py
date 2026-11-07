"""Capture the first two rank-zero calls without changing numerical statements.

Use only on an owned native WRF source/build copy. Recompile this object
with the strict scalar flags of the source build and relink its wrf.exe.
The stream uses that build's big-endian unformatted convention.
"""
from pathlib import Path
import hashlib
import json
import argparse
parser=argparse.ArgumentParser(description='Instrument an owned copy of the pinned WRF module')
parser.add_argument('module',type=Path)
parser.add_argument('--receipt',required=True,type=Path)
args=parser.parse_args()
p=args.module
s=p.read_text()
if hashlib.sha256(s.encode()).hexdigest() != '92d318297a54b3e89a2735da1bf6cfcef27b053c02339c3fb82dbbc4ca59a4cf':
    parser.error('module does not match the pinned public WRF source')
insert='''    ! Lane-only capture of actual GBMPBL bindings; numerical statements unchanged.
    integer, save :: gb_capture_count = 0
    integer :: gb_capture_unit, gb_capture_ie, gb_capture_je
    logical :: gb_capture_active
    character(len=32) :: gb_capture_rank
    character(len=80) :: gb_capture_path

    gb_capture_rank = '0'
    call get_environment_variable('OMPI_COMM_WORLD_RANK', gb_capture_rank)
    gb_capture_count = gb_capture_count + 1
    gb_capture_active = trim(gb_capture_rank) == '0' .and. gb_capture_count <= 2
    if (gb_capture_active) then
      gb_capture_ie = min(ite, its+15)
      gb_capture_je = min(jte, jts+3)
      write(gb_capture_path, '(A,I4.4,A)') 'gbm-capture-', gb_capture_count, '.bin'
      open(newunit=gb_capture_unit, file=trim(gb_capture_path), access='stream', &
           form='unformatted', status='replace', action='write')
      write(gb_capture_unit) gb_capture_ie-its+1, gb_capture_je-jts+1, kte-kts+1, dt
'''
inputs={'u':'U3D','v':'V3D','t':'T3D','qv':'QV3D','qc':'QC3D','qi':'QI3D','p':'P3D','exner':'PI3D','dz8w':'dz8w','rthraten':'RTHRATEN','tke_pbl':'TKE_PBL'}
surfaces={'psfc':'PSFC','gz1oz0':'GZ1OZ0','psim':'PSIM','xland':'XLAND','hfx':'HFX','qfx':'QFX','tsk':'TSK'}
outputs={'rublten':'RUBLTEN','rvblten':'RVBLTEN','rthblten':'RTHBLTEN','rqvblten':'RQVBLTEN','rqcblten':'RQCBLTEN','rqiblten':'RQIBLTEN','exch_m':'KZM_GBM','exch_h':'KTH_GBM','exch_tke':'KETHL_GBM','el_pbl':'EL_GBM','tke_pbl':'TKE_PBL'}
end='    if (gb_capture_active) then\n'
for name in inputs.values():insert+=f'      write(gb_capture_unit) {name}(its:gb_capture_ie,kts:kte,jts:gb_capture_je)\n'
for name in surfaces.values():insert+=f'      write(gb_capture_unit) {name}(its:gb_capture_ie,jts:gb_capture_je)\n'
insert+='    endif\n\n'
for name in outputs.values():end+=f'      write(gb_capture_unit) {name}(its:gb_capture_ie,kts:kte,jts:gb_capture_je)\n'
for name in ('PBL','UST','KPBL2D'):end+=f'      write(gb_capture_unit) {name}(its:gb_capture_ie,jts:gb_capture_je)\n'
end+='      close(gb_capture_unit)\n    endif\n'
s=s.replace('    tke2d_1 = 0.0 ',insert+'    tke2d_1 = 0.0 ',1)
s=s.replace('  END SUBROUTINE GBMPBL',end+'  END SUBROUTINE GBMPBL',1)
p.write_text(s)
args.receipt.parent.mkdir(parents=True,exist_ok=True)
args.receipt.write_text(json.dumps({'source_sha256':'92d318297a54b3e89a2735da1bf6cfcef27b053c02339c3fb82dbbc4ca59a4cf','instrumented_sha256':hashlib.sha256(s.encode()).hexdigest(),'inputs':inputs,'surfaces':surfaces,'outputs':outputs,'calls':2,'rank':0,'columns_per_call':64},indent=2)+'\n')
