"""Build the original fork timelevel routines and registry C tool under the native build prefix.

Only module storage/logging are stubbed. The five numerical/parser routines
and mpas_split_string are extracted unchanged from the pinned source.
"""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,os
REF=Path(os.environ.get('RRFS_MPAS_REF') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/MPAS-Model-rrfs')

def prepare(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    pins={}
    def routine(path,name):
        text=(REF/path).read_text()
        match=re.search(rf'^\s*subroutine {name}\(.*?^\s*end subroutine {name}\b[^\n]*',text,re.I|re.M|re.S)
        if not match:raise ValueError(name)
        window=match[0]
        pins[name]={'path':str(path),'first':text[:match.start()].count('\n')+1,'last':text[:match.end()].count('\n')+1,'sha256':hashlib.sha256(window.encode()).hexdigest()}
        return window
    names=('parse_all_timelevels','parse_time_string','parse_output_timelevel_spec','get_output_interval_from_timelevels','get_next_timelevel_start')
    timelevels = 'src/framework/mpas_output_timelevels.F'
    if not (REF/timelevels).exists():
        timelevels = 'src/framework/mpas_stream_manager.F'
    bodies=[routine(timelevels,n) for n in names]
    bodies.append(routine('src/framework/mpas_string_utils.F','mpas_split_string'))
    types=(REF/'src/framework/mpas_stream_list_types.inc').read_text().split('    type MPAS_stream_list_type')[0]
    header='''module native_timelevels
implicit none
integer,parameter :: RKIND=kind(1.0), StrKIND=512, MPAS_LOG_ERR=2, MPAS_STREAM_MGR_ERROR=-3
'''+types+'''contains
subroutine mpas_log_write(message,level,intArgs)
character(len=*),intent(in) :: message
integer,intent(in),optional :: level, intArgs(:)
end subroutine
'''
    driver='''end module
program oracle
use native_timelevels
use iso_fortran_env, only: int64
implicit none
character(len=4096) :: spec, argument
integer :: err, status, i
integer(int64) :: seconds
real :: hour, step, next, future, future_step
type(MPAS_timelevel_spec_type) :: levels
call get_command_argument(1,spec)
call get_command_argument(2,argument)
read(argument,*) seconds
call parse_all_timelevels(trim(spec),levels,'fixture',err)
if(err/=0) stop 1
do i=1,levels%n_segments
write(*,'(3(i0,1x))') transfer(levels%start_hour(i),0),transfer(levels%end_hour(i),0),transfer(levels%interval_minutes(i),0)
end do
hour=real(seconds,RKIND)/3600.0_RKIND
call get_output_interval_from_timelevels(levels,hour,step,status,next)
call get_next_timelevel_start(levels,hour,future,future_step,err)
write(*,'(6(i0,1x))') status,transfer(hour,0),transfer(step,0),transfer(next,0),transfer(future,0),transfer(future_step,0)
end program
'''
    (destination/'native_rrfs_timelevels.f90').write_text(header+'\n'.join(bodies)+'\n'+driver)
    (destination/'native-source-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    registry=destination/'registry-tool'
    shutil.copytree(REF/'src/tools/registry',registry,dirs_exist_ok=True)
    shutil.copytree(REF/'src/external/ezxml',destination/'ezxml',dirs_exist_ok=True)
    shutil.copytree(REF/'src/core_atmosphere',destination/'registry-input',ignore=shutil.ignore_patterns('*.F','*.f90','*.F90','physics_wrf','physics_mmm','physics_noaa','dynamics'),dirs_exist_ok=True)
    shutil.copyfile(REF/'src/framework/xml_stream_parser.c', destination/'xml_stream_parser.c')
    shutil.copyfile(Path(__file__).with_name('native_rrfs_xml_recorder.c'),destination/'native_rrfs_xml_recorder.c')
    shutil.copytree(REF/'src/tools/input_gen',destination/'input-gen',dirs_exist_ok=True)
    makefile = r"""FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form
all: native_rrfs_timelevels native_rrfs_xml registry-full registry-dry defaults
native_rrfs_timelevels: native_rrfs_timelevels.f90
	gfortran $(FLAGS) $< -o $@
native_rrfs_xml: xml_stream_parser.c native_rrfs_xml_recorder.c ezxml/ezxml.c
	gcc -O0 -g -Iezxml $^ -o $@
registry-tool/registry: registry-tool/parse.c registry-tool/gen_inc.c registry-tool/dictionary.c registry-tool/fortprintf.c registry-tool/utility.c ezxml/ezxml.c
	gcc -O0 -g -Iezxml $^ -o $@
registry-full: registry-tool/registry
	mkdir -p native-registry/full
	cpp -P -traditional -DDO_PHYSICS registry-input/Registry.xml > native-registry/full/Registry_processed.xml
	cd native-registry/full && ../../registry-tool/registry Registry_processed.xml -DSINGLE_PRECISION
registry-dry: registry-tool/registry
	mkdir -p native-registry/dry
	cpp -P -traditional registry-input/Registry.xml > native-registry/dry/Registry_processed.xml
	cd native-registry/dry && ../../registry-tool/registry Registry_processed.xml -DSINGLE_PRECISION; status=$$?; echo native_dry_registry_exit=$$status; test -f structs_and_variables.inc && test -f setup_immutable_streams.inc
input-gen/streams_gen: input-gen/streams_gen.c input-gen/test_functions.c ezxml/ezxml.c
	gcc -O0 -g -Iezxml $^ -o $@
defaults: registry-full input-gen/streams_gen
	mkdir -p native-defaults
	cd native-defaults && ../input-gen/streams_gen ../native-registry/full/Registry_processed.xml streams.atmosphere stream_list.atmosphere. listed in_defaults=true
"""
    (destination/'Makefile.lane').write_text(makefile)
    return pins

if __name__=='__main__':
    import sys
    prepare(sys.argv[1])
