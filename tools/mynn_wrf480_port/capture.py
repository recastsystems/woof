#!/usr/bin/env python3
"""Insert I/O-only, opt-in capture hooks in a private WRF 4.8.0 tree.

Edits only phys/MYNN-EDMF/module_bl_mynnedmf.F90 of a private copy; the
referee binary is built from the untouched tree. Hooks write every core
argument at entry and exit of mynnedmf for a fixed sample of columns
(i, j congruent to 12 mod 25) on PBL calls 1, 2, 3, 10, 30 and 60 of each
sampled column, so captures include spun-up states, not only the cold start.
"""
from pathlib import Path
import argparse,json,re
from lower import read_units,schema,decl
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
f=a.source/'phys/MYNN-EDMF/module_bl_mynnedmf.F90';src=f.read_text();unit=read_units(f)[0][0];fields=schema(unit)
args=[n for n,d in fields.items() if d['type']!='character']
# 8f8e8b59 declares the TKE-budget arrays optional and WRF 4.8.0 passes them
# unallocated (absent) unless tke_budget=1. Write a zero column in their
# place so every capture has one fixed layout; the core touches them only
# under tke_budget=1, so a present zero column replays the same words.
optional=set()
for s_ in unit['body']:
    d=decl(s_)
    if d and 'optional' in d[1]:
        for name,_,_ in d[2]: optional.add(name)
for name in optional:
    assert fields[name]['dims']=='kts:kte',(name,fields[name])
def put(n):
    nl=chr(10)
    if n in optional:
        return ('      if (present('+n+')) then'+nl+'        write(capture_unit) '+n+nl+
                '      else'+nl+'        write(capture_unit) capture_zero'+nl+'      endif'+nl)
    return '      write(capture_unit) '+n+nl
declarations='''
! I/O-only real-input replay capture. No numerical statements changed.
 integer, save :: capture_calls(0:255,0:255)=0
 integer :: capture_unit,capture_stat,capture_n
 character(len=1024) :: capture_root,capture_name
 logical :: capture_this
 real(kind_phys) :: capture_zero(kts:kte)
'''
start='    errmsg = " "'
assert src.count(start)==1
before='''
    capture_this=.false.
    capture_zero=0.
    call get_environment_variable('WRF_MYNN_CAPTURE_DIR',capture_root,status=capture_stat)
    if (capture_stat==0 .and. mod(i,25)==12 .and. mod(j,25)==12 .and. i/25<=255 .and. j/25<=255) then
      capture_calls(i/25,j/25)=capture_calls(i/25,j/25)+1
      capture_n=capture_calls(i/25,j/25)
      if (capture_n==1 .or. capture_n==2 .or. capture_n==3 .or. capture_n==10 .or. &
          capture_n==30 .or. capture_n==60) then
      capture_this=.true.
      write(capture_name,'(a,a,i4.4,a,i4.4,a,i4.4,a)') &
        trim(capture_root),'/column-',i,'-',j,'-',capture_n,'.input.bin'
      open(newunit=capture_unit,file=trim(capture_name),access='stream', &
        form='unformatted',status='replace')
'''+''.join(put(n) for n in args)+'''      close(capture_unit)
      endif
    endif
'''
after='''
    if (capture_this) then
      write(capture_name,'(a,a,i4.4,a,i4.4,a,i4.4,a)') &
        trim(capture_root),'/column-',i,'-',j,'-',capture_n,'.output.bin'
      open(newunit=capture_unit,file=trim(capture_name),access='stream', &
        form='unformatted',status='replace')
'''+''.join(put(n) for n in args)+'''      close(capture_unit)
    endif
'''
src=src.replace(start,declarations+before+'\n'+start)
src, count = re.subn(r'(?im)^\s*end subroutine mynnedmf\s*$',
                    lambda m: after+'\n'+m[0], src, count=1)
assert count == 1, 'Native MYNN end marker was not found'
f.write_text(src);(a.source/'mynn-capture-arguments.json').write_text(json.dumps(fields,indent=2)+'\n')
