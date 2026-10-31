"""Build a strict standalone oracle from MPAS's own clock sources, read-only.

Usage: python build_native_stream_clock.py <owned scratch directory>
Only logging, threading and the enclosing clock storage are framework stubs;
interval parsing, calendar arithmetic, filename division and alarms are native.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import re
import shutil
import subprocess
import sys
BUILD_PREFIX = shlex.split(os.environ.get('HEX_NATIVE_BUILD_PREFIX', ''))  # the host's bounded build wrapper, if any


def build(destination: Path) -> Path:
    reference = Path(os.environ.get('MPAS_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/MPAS-Model-v8.4.2')
    destination.mkdir(parents=True, exist_ok=True)
    esmf = destination/'esmf'
    if esmf.exists():
        esmf.chmod(0o755)
        for directory in esmf.rglob('*'):
            if directory.is_dir():
                directory.chmod(0o755)
    def copy_source(source, target):
        target = Path(target)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Native build copy must remain in owned scratch')
        if target.exists():
            target.chmod(0o644)
        return shutil.copyfile(source, target)
    shutil.copytree(reference/'src/external/esmf_time_f90', esmf,
                    dirs_exist_ok=True, copy_function=copy_source)
    # copytree preserves the reference's read-only directory mode too.
    esmf.chmod(0o755)
    for directory in esmf.rglob('*'):
        if directory.is_dir():
            directory.chmod(0o755)
    flags = '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fconvert=big-endian -ffree-form'
    cpp_flags = '-cpp -DHIDE_MPI -DSINGLE_PRECISION'
    subprocess.run([*BUILD_PREFIX,'taskset','-c','8-23','make','-j4','FC=gfortran',
                    f'CPPFLAGS={cpp_flags}',f'FFLAGS={flags}'], cwd=esmf, check=True)
    pins = {}

    def routine(path, name):
        text = (reference/path).read_text()
        pattern = rf'^\s*(?:logical function|subroutine) {name}\(.*?^\s*end (?:function|subroutine) {name}\b[^\n]*'
        match = re.search(pattern, text, flags=re.I|re.M|re.S)
        if not match:
            raise ValueError(f'Pinned source lacks routine {name}')
        window = match.group(0)
        pins[name] = {'source_path':path, 'first_line':text[:match.start()].count('\n')+1,
                      'last_line':text[:match.end()].count('\n')+1,
                      'window_sha256':hashlib.sha256(window.encode()).hexdigest()}
        # Substitute native wrapper storage with the underlying ESMF type.
        window = re.sub(r'MPAS_TimeInterval_type', 'ESMF_TimeInterval', window, flags=re.I)
        window = re.sub(r'MPAS_Time_type', 'ESMF_Time', window, flags=re.I)
        window = re.sub(r'\s*%\s*ti\b', '', window, flags=re.I)
        window = re.sub(r'^\s*STREAM_DEBUG_WRITE\([^\n]*', '', window, flags=re.M)
        return window

    clock_source = 'src/framework/mpas_timekeeping.F'
    bodies = [routine(clock_source, name) for name in
              ('mpas_set_timeInterval','mpas_interval_division','mpas_add_clock_alarm',
               'mpas_adjust_alarm_to_reference_time','mpas_in_ringing_envelope','mpas_reset_clock_alarm')]
    bodies.append(routine('src/framework/mpas_string_utils.F','mpas_split_string'))
    bodies.append(routine('src/framework/mpas_stream_manager.F','mpas_build_stream_filename'))
    header = '''module native_stream_clock
use ESMF
implicit none
integer, parameter :: I8KIND=8, RKIND=4, StrKIND=512, ShortStrKIND=64
integer, parameter :: MPAS_LOG_ERR=1, MPAS_LOG_WARN=2, MPAS_LOG_CRIT=3, MPAS_FORWARD=1, MPAS_NOW=1
type MPAS_Alarm_type
  character(len=64) :: alarmID
  logical :: isSet, isRecurring
  type(ESMF_Time) :: ringTime, prevRingTime
  type(ESMF_TimeInterval) :: ringTimeInterval
  type(MPAS_Alarm_type), pointer :: next => null()
end type
type MPAS_Clock_type
  type(ESMF_Time) :: now
  integer :: direction=MPAS_FORWARD, nAlarms=0
  type(MPAS_Alarm_type), pointer :: alarmListHead => null()
end type
contains
integer function mpas_threading_get_thread_num()
mpas_threading_get_thread_num=0
end function
subroutine mpas_log_write(message, level, intArgs)
character(len=*), intent(in) :: message
integer, optional :: level, intArgs(:)
write(0,'(a)') trim(message)
end subroutine
function mpas_get_clock_time(clock, selector, ierr) result(now)
type(MPAS_Clock_type), intent(in) :: clock
integer :: selector
integer, optional :: ierr
type(ESMF_Time) :: now
now=clock%now
if (present(ierr)) ierr=0
end function
subroutine mpas_get_time(now, dateTimeString)
type(ESMF_Time), intent(in) :: now
character(len=*), intent(out) :: dateTimeString
call ESMF_TimeGet(now, TimeString=dateTimeString)
end subroutine
subroutine mpas_get_timeInterval(interval, timeString)
type(ESMF_TimeInterval), intent(in) :: interval
character(len=*), intent(out) :: timeString
call ESMF_TimeIntervalGet(interval, TimeString=timeString)
end subroutine
subroutine mpas_expand_string(stamp, block, template, filename)
character(len=*), intent(in) :: stamp, template
integer, intent(in) :: block
character(len=*), intent(out) :: filename
filename=stamp
end subroutine
subroutine parse_time(text, now)
character(len=*), intent(in) :: text
type(ESMF_Time), intent(out) :: now
integer :: yy, mm, dd, h, m, s, sn, sd, count
read(text(1:19),'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)') yy, mm, dd, h, m, s
sn=0
sd=1
if (len_trim(text)>20) then
  count=min(len_trim(text)-20,8)
  read(text(21:20+count),*) sn
  sd=10**count
endif
call ESMF_TimeSet(now, YY=yy, MM=mm, DD=dd, H=h, M=m, S=s, Sn=sn, Sd=sd, calkindflag=ESMF_CALKIND_GREGORIAN)
end subroutine
end module
'''
    # The driver only supplies time columns and requests native routines.
    driver = '''program clock_columns
use native_stream_clock
implicit none
type(MPAS_Clock_type) :: clock
type(ESMF_Time) :: origin, start, when, advanced
type(ESMF_TimeInterval) :: period
character(len=512) :: text, filename, previous, added
integer :: count, i, reset, ierr, pn, pd, an, ad
logical :: ringing
call ESMF_Initialize(defaultCalendar=ESMF_CALKIND_GREGORIAN)
read(*,'(a)') text
call parse_time(text,origin)
read(*,'(a)') text
call parse_time(text,start)
read(*,'(a)') text
call mpas_set_timeInterval(period,timeString=trim(text),ierr=ierr)
if (ierr/=0) stop 2
clock%now=start
call mpas_add_clock_alarm(clock,'fixture',start,period,ierr)
call mpas_adjust_alarm_to_reference_time(clock,'fixture',origin,ierr)
read(*,*) count
do i=1,count
  read(*,'(a)') text
  call parse_time(text,when)
  read(*,*) reset
  clock%now=when
  call mpas_build_stream_filename(origin,when,period,'identity',0,filename,ierr)
  ringing=mpas_in_ringing_envelope(clock,clock%alarmListHead)
  if (reset/=0) call mpas_reset_clock_alarm(clock,'fixture',ierr=ierr)
  call mpas_get_time(clock%alarmListHead%prevRingTime,dateTimeString=previous)
  call ESMF_TimeGet(clock%alarmListHead%prevRingTime,Sn=pn,Sd=pd)
  advanced=when+period
  call mpas_get_time(advanced,dateTimeString=added)
  call ESMF_TimeGet(advanced,Sn=an,Sd=ad)
  write(*,'(a,"|",l1,"|",a,"|",a,4("|",i0))') trim(filename),ringing,trim(previous),trim(added),pn,pd,an,ad
enddo
call ESMF_Finalize()
end program
'''
    module_text = header.replace('end module\n', '\n'.join(bodies)+'\nend module\n')
    source = destination/'oracle.f90'
    source.write_text(module_text+driver)
    binary = destination/'native_stream_clock'
    subprocess.run([*BUILD_PREFIX,'taskset','-c','8-23','gfortran',*flags.split(),
                    '-I',str(esmf),str(source),str(esmf/'libesmf_time.a'),'-o',str(binary)],
                   cwd=destination,check=True)
    (destination/'clock-build.json').write_text(json.dumps({
        'schema':'mpas-native-stream-clock-build/v1','compiler_flags':flags,
        'cpp_flags':cpp_flags,'build_host':os.uname().nodename,
        'compiler':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        'oracle_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'generated_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_windows':pins,'native_process_count':1,'transport_process_count':1,
        'whole_model_proof':False},indent=2)+'\n')
    return binary


if __name__ == '__main__':
    print(build(Path(sys.argv[1]).resolve()))
