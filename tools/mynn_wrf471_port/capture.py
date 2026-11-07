#!/usr/bin/env python3
"""Insert I/O-only, opt-in capture hooks in a private WRF 4.7.1 tree."""
from pathlib import Path
import argparse,json,re
from lower import read_units,schema
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
f=a.source/'phys/module_bl_mynnedmf.F';src=f.read_text();unit=read_units(f)[0][0];fields=schema(unit)
args=[n for n,d in fields.items() if d['type']!='character']
declarations='''
! I/O-only real-input replay capture. No numerical statements changed.
 integer, save :: capture_count=0
 integer :: capture_unit,capture_stat
 character(len=1024) :: capture_root,capture_name
 logical :: capture_this
'''
start='    errmsg = " "'
assert src.count(start)==1
before='''
    capture_this=.false.
    call get_environment_variable('WRF_MYNN_CAPTURE_DIR',capture_root,status=capture_stat)
    if (capture_stat==0 .and. mod(i,50)==25 .and. mod(j,50)==25 .and. capture_count<36) then
      capture_count=capture_count+1
      capture_this=.true.
      write(capture_name,'(a,a,i4.4,a,i4.4,a,i4.4,a)') &
        trim(capture_root),'/column-',i,'-',j,'-',capture_count,'.input.bin'
      open(newunit=capture_unit,file=trim(capture_name),access='stream', &
        form='unformatted',status='replace')
'''+''.join('      write(capture_unit) '+n+'\n' for n in args)+'''      close(capture_unit)
    endif
'''
after='''
    if (capture_this) then
      write(capture_name,'(a,a,i4.4,a,i4.4,a,i4.4,a)') &
        trim(capture_root),'/column-',i,'-',j,'-',capture_count,'.output.bin'
      open(newunit=capture_unit,file=trim(capture_name),access='stream', &
        form='unformatted',status='replace')
'''+''.join('      write(capture_unit) '+n+'\n' for n in args)+'''      close(capture_unit)
    endif
'''
src=src.replace(start,declarations+before+'\n'+start)
src, count = re.subn(r'(?im)^\s*end subroutine mynnedmf\s*$',
                    lambda m: after+'\n'+m[0], src, count=1)
assert count == 1, 'Native MYNN end marker was not found'
f.write_text(src);(a.source/'mynn-capture-arguments.json').write_text(json.dumps(fields,indent=2)+'\n')
