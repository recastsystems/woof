"""Strict MPAS_seekStream oracle; MPAS mpas_io_streams.F:153-339.

Reuse the existing native ESMF/timekeeping build. Only file storage/PIO access
is stubbed: timestamps are supplied verbatim, and the entire pinned seek routine
and native calendar/interval arithmetic determine record selection.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import re
import subprocess
import sys

from build_native_stream_clock import build as build_clock
BUILD_PREFIX = shlex.split(os.environ.get('HEX_NATIVE_BUILD_PREFIX', ''))  # the host's bounded build wrapper, if any


def build(destination):
    """Compile the whole seek routine; MPAS mpas_io_streams.F:153-339."""
    destination=Path(destination).resolve()
    build_clock(destination)
    reference=Path(os.environ.get('MPAS_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/MPAS-Model-v8.4.2')
    path='src/framework/mpas_io_streams.F'
    source=(reference/path).read_text()
    match=re.search(r'^\s*integer function MPAS_seekStream\(.*?^\s*end function MPAS_seekStream\b[^\n]*',
                    source,re.I|re.M|re.S)
    assert match, 'pinned whole seek routine required'
    body=match.group(0)
    body=re.sub(r'MPAS_TimeInterval_type','ESMF_TimeInterval',body,flags=re.I)
    body=re.sub(r'MPAS_Time_type','ESMF_Time',body,flags=re.I)
    time_source=(reference/'src/framework/mpas_timekeeping.F').read_text()
    abs_match=re.search(r'^\s*type \(MPAS_TimeInterval_type\) function abs_ti\(.*?^\s*end function abs_ti\b[^\n]*',
                        time_source,re.I|re.M|re.S)
    assert abs_match, 'native absolute interval wrapper required'
    absolute=re.sub(r'MPAS_TimeInterval_type','ESMF_TimeInterval',abs_match.group(0),flags=re.I)
    absolute=re.sub(r'\s*%\s*ti\b','',absolute,flags=re.I)
    module=(destination/'oracle.f90').read_text().split('end module\n',1)[0]
    module=module.replace('contains\n', '''integer, parameter :: MPAS_IO_NOERR=0, MPAS_IO_ERR=-5, MPAS_STREAM_NOERR=0
integer, parameter :: MPAS_STREAM_EXACT_TIME=100, MPAS_STREAM_NEAREST=101, &
  MPAS_STREAM_LATEST_BEFORE=102, MPAS_STREAM_EARLIEST_AFTER=103, &
  MPAS_STREAM_LATEST_STRICTLY_BEFORE=104, MPAS_STREAM_EARLIEST_STRICTLY_AFTER=105
type MPAS_Stream_type
  logical :: isInitialized=.true.
  integer :: fileHandle=1
end type
character(len=64), allocatable :: file_times(:)
integer :: current_frame=1
interface abs
  module procedure abs_ti
end interface
contains
''',1)
    stubs='''
subroutine MPAS_io_inq_dim(handle,name,length,ierr)
integer,intent(in)::handle
character(len=*),intent(in)::name
integer,intent(out)::length,ierr
length=size(file_times)
ierr=0
end subroutine
subroutine MPAS_io_inq_var(handle,name,ierr)
integer,intent(in)::handle
character(len=*),intent(in)::name
integer,intent(out)::ierr
ierr=0
end subroutine
subroutine MPAS_io_set_frame(handle,frame,ierr)
integer,intent(in)::handle,frame
integer,intent(out)::ierr
current_frame=frame
ierr=0
end subroutine
subroutine MPAS_io_get_var(handle,name,value,ierr)
integer,intent(in)::handle
character(len=*),intent(in)::name
character(len=*),intent(out)::value
integer,intent(out)::ierr
value=file_times(current_frame)
ierr=0
end subroutine
subroutine mpas_set_time(curr_time,dateTimeString)
type(ESMF_Time),intent(out)::curr_time
character(len=*),intent(in)::dateTimeString
call parse_time(dateTimeString,curr_time)
end subroutine
'''
    driver='''
program seek_columns
use native_stream_clock
implicit none
type(MPAS_Stream_type)::stream
character(len=64)::target,actual
integer::count,mode,i,index,ierr,maxRecords
call ESMF_Initialize(defaultCalendar=ESMF_CALKIND_GREGORIAN)
read(*,*)count,mode
allocate(file_times(count))
read(*,'(a)')target
do i=1,count
  read(*,'(a)')file_times(i)
enddo
index=MPAS_seekStream(stream,trim(target),mode,actual,maxRecords,ierr)
if(index>0)then
  write(*,'(i0,"|",i0,"|",i0,"|",a)')index,ierr,maxRecords,trim(actual)
else
  write(*,'(i0,"|",i0,"|",i0)')index,ierr,maxRecords
endif
call ESMF_Finalize()
end program
'''
    generated=destination/'seek-oracle.f90'
    generated.write_text(module+stubs+absolute+body+'\nend module\n'+driver)
    binary=destination/'native_stream_seek'
    flags=json.loads((destination/'clock-build.json').read_text())['compiler_flags']
    subprocess.run([*BUILD_PREFIX,'taskset','-c','8-23','gfortran',*flags.split(),
        '-I',str(destination/'esmf'),str(generated),str(destination/'esmf/libesmf_time.a'),
        '-o',str(binary)],cwd=destination,check=True)
    (destination/'seek-build.json').write_text(json.dumps(dict(
        schema='o1-native-stream-seek-build-v1',process_count=1,build_host=os.uname().nodename,
        compiler_flags=flags,source_path=path,first_line=source[:match.start()].count('\n')+1,
        last_line=source[:match.end()].count('\n')+1,
        native_window_sha256=hashlib.sha256(match.group(0).encode()).hexdigest(),
        native_abs_wrapper_sha256=hashlib.sha256(abs_match.group(0).encode()).hexdigest(),
        native_abs_wrapper_source='src/framework/mpas_timekeeping.F:2085-2107',
        generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        scope='Whole pinned seek routine and native ESMF calendar; PIO timestamp storage stub only'),indent=2)+'\n')
    return binary


if __name__=='__main__':
    print(build(Path(sys.argv[1])))
