"""Fork source-family selection and actual compiled nested oracle semantics."""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import subprocess
import sys

import pytest

from woof.hex import site_runner

ORACLE=Path(__file__).resolve().parents[1]/'tools/native_physics_reference/oracle'
sys.path.insert(0,str(ORACLE))
from pin_sets import source_files,pin_set,reference_build
from derived import primitive_components
from generate_transport import generate
from fortran_source import EXECUTABLE,first_executable,Statement,Variable
import site_support
from test_native_physics_oracle import transport, _build_command


def test_oracle_registry_metadata_matches_audited_native_build(tmp_path):
    from prepare_build import bind_reference_build_metadata
    build=tmp_path/'build';build.mkdir()
    reference=tmp_path/'reference';reference.mkdir()
    makefile=build/'Makefile'
    original='''before
VER=$(shell git describe --dirty 2> /dev/null)
#override CPPFLAGS += -DMPAS_GIT_VERSION=$(VER)
ifeq "$(findstring v, $(VER))" "v"
 override CPPFLAGS += -DMPAS_GIT_VERSION=$(VER)
else
 override CPPFLAGS += -DMPAS_GIT_VERSION="unknown"
endif # END OF GIT DESCRIBE VERSION
after
'''
    makefile.write_text(original)
    audit=reference/'flag-audit.json'
    audit.write_text(json.dumps(dict(all_ok=True,recipe='gnu-strict',objects=[dict(commands=[
        dict(argv=['-DMPAS_GIT_VERSION=unknown','-O3'])])])) )
    binding=bind_reference_build_metadata(build,reference)
    assert binding==dict(git_version='unknown',audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
    assert makefile.read_text().startswith('before\n') and makefile.read_text().endswith('\nafter\n')
    assert 'git describe' not in makefile.read_text()
    assert 'override CPPFLAGS += -DMPAS_GIT_VERSION=unknown' in makefile.read_text()
    makefile.write_text(original)
    audit.write_text(json.dumps(dict(all_ok=True,recipe='gnu-strict',objects=[dict(commands=[
        dict(argv=['-DMPAS_GIT_VERSION=unknown','-DMPAS_GIT_VERSION=v8.dirty'])])])))
    with pytest.raises(ValueError,match='one Registry git version'):
        bind_reference_build_metadata(build,reference)
    assert makefile.read_text()==original


def test_passivity_variant_restores_complete_native_bodies(tmp_path):
    sys.path.insert(0, str(ORACLE/'replay'))
    from prepare_passivity_variant import restore_bodies
    native = tmp_path/'native.F90'
    generated = tmp_path/'instrumented.F90'
    body = '''subroutine compute(x)
 real,intent(inout) :: x
 if(x<0) return
 x=x*2
end subroutine compute
'''
    native.write_text('module physics\ncontains\n'+body+'end module physics\n')
    generated.write_text('module physics\ncontains\n'+body.replace(
        ' x=x*2', ' call intrusive_hook(x)\n x=x*2')+
        'subroutine oracle_state_physics()\nend subroutine oracle_state_physics\nend module physics\n')
    restored, rows = restore_bodies(native, generated, {'physics'})
    assert body in restored and 'intrusive_hook' not in restored
    assert 'subroutine oracle_state_physics()' in restored
    assert len(rows)==1 and rows[0]['source_sha256']==hashlib.sha256(body.encode()).hexdigest()
    generated.write_text('module physics\nend module physics\n')
    with pytest.raises(ValueError, match='native routine absent'):
        restore_bodies(native, generated, {'physics'})


def test_passivity_relink_uses_only_exact_sealed_parent_source_snapshots(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from prepare_passivity_variant import parent_dependencies
    original=tmp_path/'live.F90';original.write_text('native original source')
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    parent=tmp_path/'parent';parent.mkdir()
    snapshot=parent/'physics.F90';snapshot.write_bytes(original.read_bytes())
    manifest=dict(dependencies={str(original):digest},rows=[dict(path='src/physics.F90',
        source_path=str(original),source_sha256=digest,variant_sha256=digest)])
    assert parent_dependencies(parent,manifest)==([original],[])
    original.write_text('later source edit')
    paths,archived=parent_dependencies(parent,manifest)
    assert paths==[snapshot] and archived==[dict(original_path=str(original),snapshot_path=str(snapshot),sha256=digest)]
    snapshot.write_text('different archived source')
    with pytest.raises(ValueError,match='source snapshot changed'):
        parent_dependencies(parent,manifest)
    manifest['rows']=[]
    with pytest.raises(ValueError,match='build dependency changed'):
        parent_dependencies(parent,manifest)


def test_passivity_bodyless_module_retains_active_caller_serializers(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from prepare_passivity_variant import restore_bodies
    original=tmp_path/'ESMF.F90';generated=tmp_path/'generated.F90'
    original.write_text('module ESMF\n use ESMF_TimeMod\nend module ESMF\n')
    generated.write_text('''module ESMF
 use ESMF_TimeMod
 public :: oracle_state_esmf
contains
 subroutine oracle_state_esmf()
 end subroutine oracle_state_esmf
end module ESMF
''')
    restored,rows=restore_bodies(original,generated,set())
    assert restored==generated.read_text() and rows==[]


def test_passivity_variant_seals_complete_compiled_module(tmp_path):
    sys.path.insert(0, str(ORACLE/'replay'))
    from prepare_passivity_variant import seal_variant
    from run_passivity_variant import authenticate_variant
    source = tmp_path/'physics.F90'
    source.write_text('module physics\nend module physics\n')
    recipe = tmp_path/'Makefile'
    recipe.write_text('all: atmosphere_model\n')
    dependency = tmp_path/'frozen.a'
    dependency.write_bytes(b'original library')
    for name in ('physics.o', 'libdycore.a', 'atmosphere_model'):
        (tmp_path/name).write_bytes(name.encode())
    manifest = dict(rows=[dict(path='physics.F90', variant_sha256=hashlib.sha256(source.read_bytes()).hexdigest())],
        dependencies={str(dependency):hashlib.sha256(dependency.read_bytes()).hexdigest()},
        makefile_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest())
    (tmp_path/'variant.json').write_text(json.dumps(manifest))
    seal_variant(tmp_path)
    seal = tmp_path/'artifacts.json'
    assert authenticate_variant(tmp_path/'atmosphere_model',seal)==hashlib.sha256(seal.read_bytes()).hexdigest()
    (tmp_path/'physics.o').write_bytes(b'changed object')
    with pytest.raises(ValueError,match='artifact changed'):
        authenticate_variant(tmp_path/'atmosphere_model',seal)
    source.write_text('changed source')
    with pytest.raises(ValueError,match='native body changed'):
        seal_variant(tmp_path)


def test_passivity_variant_restores_native_private_procedures(tmp_path):
    sys.path.insert(0, str(ORACLE/'replay'))
    from prepare_passivity_variant import restore_private_procedures
    native = tmp_path/'native.F90'
    native.write_text('''module physics
 private
 public :: compute
contains
 subroutine compute()
 end subroutine
 subroutine helper()
 end subroutine
end module
''')
    generated = native.read_text().replace('contains',
        'public :: helper,oracle_state_physics\ncontains')
    result, changed = restore_private_procedures(native, generated, {'physics'})
    assert changed=={'physics':['helper']}
    assert 'public :: compute' in result and 'public :: oracle_state_physics' in result
    assert 'public :: helper' not in result
    assert result[result.index('contains'):]==generated[generated.index('contains'):]


def test_passivity_variant_rejects_unbound_sources_and_wrong_library(tmp_path):
    sys.path.insert(0, str(ORACLE/'replay'))
    from prepare_passivity_variant import prepare
    build=tmp_path/'build';build.mkdir()
    (build/'oracle-inventory.json').write_text(json.dumps(dict(
        module_state=[dict(name='framework',path='src/framework/framework.F')],
        generated_sha256={})))
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(all_ok=True,recipe='gnu-strict',objects=[])))
    out=tmp_path/'variant'
    arguments=(build,tmp_path,audit,tmp_path/'link.log')
    with pytest.raises(ValueError,match='restricted to the dycore library'):
        prepare(*arguments,['framework'],out)
    with pytest.raises(ValueError,match='frozen generated member'):
        prepare(*arguments,[],out,source_paths=['../unbound.F'])
    assert not out.exists()


def test_disabled_nested_native_tap_does_not_allocate_counter_storage(transport,tmp_path):
    """Disabled instrumentation must not perturb large native heap allocations."""
    (tmp_path/'transport.F').write_text(generate())
    (tmp_path/'heap.c').write_text('''#include <stddef.h>
extern void *__real_malloc(size_t);
static int active, calls;
void *__wrap_malloc(size_t bytes) {
    if (active && bytes >= 16384) ++calls;
    return __real_malloc(bytes);
}
void begin_heap_probe(void) { calls = 0; active = 1; }
int end_heap_probe(void) { active = 0; return calls; }
''')
    (tmp_path/'fixture.F90').write_text('''program disabled_scopes
 use iso_c_binding
 use mpas_oracle_dump
 implicit none
 interface
  subroutine begin_heap_probe() bind(c)
  end subroutine
  integer(c_int) function end_heap_probe() bind(c)
   import c_int
  end function
 end interface
 integer :: event, iteration
 integer(c_int32_t),target :: native_ids(-7:10232)
 integer(c_int32_t),pointer :: native_id_pointer(:)
 character(len=4096) :: output_prefix
 native_ids=7
 native_id_pointer=>native_ids
 call begin_heap_probe()
 do iteration=1,8
  call oracle_set_native_context(native_id_pointer,iteration,0)
  call oracle_begin('inactive_outer',event)
  call oracle_scope_begin('inactive_outer',event)
  call oracle_begin('inactive_inner',event)
  call oracle_scope_begin('inactive_inner',event)
  call oracle_begin('inactive_leaf',event)
  call oracle_scope_begin('inactive_leaf',event)
  call oracle_scope_end()
  call oracle_scope_end()
  call oracle_scope_end()
 enddo
 if(end_heap_probe()/=0) error stop 'disabled scopes perturb native heap'
 ! Restoring an owned context must never free borrowed native storage.
 call oracle_set_context([101_c_int32_t,205_c_int32_t],1,0)
 if(any(native_ids/=7)) error stop 'borrowed native context modified'
 call get_command_argument(1,output_prefix)
 call oracle_set_output(trim(output_prefix))
 call oracle_set_native_context(native_id_pointer,2,0)
 native_ids(-7:-6)=[71_c_int32_t,83_c_int32_t]
 call oracle_write('borrowed','post','ids',native_ids(-7:-6),1,cell_axis=1)
end program
''')
    (tmp_path/'Makefile').write_text('all: fixture\nheap.o: heap.c\n\tgcc -O3 -c $< -o $@\nfixture: transport.F fixture.F90 heap.o\n\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form $^ -Wl,--wrap=malloc -o $@\n')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,text=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MPAS_ORACLE_')}
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path/'borrowed')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(transport/'oracle_words'),'header',str(tmp_path/'borrowed.post.ids.bin')],
                          check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['global_ids']==[71,83]
    legacy=subprocess.run([str(transport/'oracle_words'),'header',str(tmp_path/'borrowed.post.ids.bin'),'-'],
                          check=True,capture_output=True,text=True)
    assert legacy.stdout==result.stdout
    for arguments in ([],['compare'],['compare',str(tmp_path/'borrowed.post.ids.bin')],
                      ['header',str(tmp_path/'borrowed.post.ids.bin'),'unexpected']):
        invalid=subprocess.run([str(transport/'oracle_words')]+arguments,capture_output=True,text=True)
        assert invalid.returncode!=0 and 'usage:' in invalid.stderr and 'panicked' not in invalid.stderr


def test_native_pool_replay_retains_cross_pool_storage_and_rejects_wrong_context(transport,tmp_path):
    """Round-trip the actual initialized native graph, including shared storage."""
    import shutil
    from generate_pool import generate as pool_source
    from generate_pool_restore import generate as restore_source
    import test_native_physics_oracle as native_tests
    build=os.environ.get('MPAS_ORACLE_NATIVE_BUILD')
    if not build:
        pytest.skip('set MPAS_ORACLE_NATIVE_BUILD to an owned strict MPAS build')
    build=Path(build)
    for p in (build/'src/framework').glob('*.mod'):
        shutil.copy2(p,tmp_path/p.name)
    shutil.copy2(build/'src/framework/libframework.a',tmp_path/'libframework.a')
    shutil.copy2(build/'src/external/esmf_time_f90/libesmf_time.a',tmp_path/'libesmf_time.a')
    (tmp_path/'transport.F').write_text(generate())
    (tmp_path/'pool.F').write_text(pool_source())
    (tmp_path/'restore.F').write_text(restore_source())
    shutil.copy2(ORACLE/'replay/pool_fixture.F90',tmp_path/'fixture.F90')
    (tmp_path/'Makefile').write_text('''FC=mpif90
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian
all: fixture
transport.o: transport.F
\t$(FC) $(FLAGS) -c $< -o $@
pool.o: pool.F transport.o
\t$(FC) $(FLAGS) -c $< -o $@
restore.o: restore.F transport.o
\t$(FC) $(FLAGS) -c $< -o $@
fixture: fixture.F90 pool.o restore.o
\t$(FC) $(FLAGS) $^ transport.o -Wl,--start-group libframework.a libesmf_time.a -Wl,--end-group -lnetcdff -lnetcdf -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,text=True)
    def run(directory,mode=None):
        directory.mkdir()
        args=[str(tmp_path/'fixture')]+[str(directory/name) for name in ('original','fresh','in_place')]
        if mode:args.append(mode)
        return subprocess.run(args,capture_output=True,text=True)
    directory=tmp_path/'valid'
    result=run(directory)
    assert result.returncode==0,result.stdout+result.stderr
    header=subprocess.run([str(transport/'oracle_words'),'header',
        str(directory/'original.pre.p.offset_dimension.bin')],check=True,capture_output=True,text=True)
    assert json.loads(header.stdout)['lower']==[-2]
    originals={p.name.removeprefix('original.post.'):p for p in directory.glob('original.post.*.bin')}
    for mode in ('fresh','in_place'):
        actual={p.name.removeprefix(mode+'.post.'):p for p in directory.glob(mode+'.post.*.bin')}
        assert actual.keys()==originals.keys() and len(actual)>100
        for name,p in originals.items():
            assert json.loads(native_tests.compare(transport,p,actual[name]).stdout)['bitwise_equal'],(mode,name)
    result=run(tmp_path/'wrong_context','bad-bounds')
    assert result.returncode!=0 and 'context bounds mismatch' in result.stdout+result.stderr


def test_source_selection_excludes_other_model_wrappers(tmp_path):
    physics=tmp_path/'src/core_atmosphere/physics/physics_noaa/MYNN-SFC'
    for directory in ('MPAS','WRF','CCPP','docs/src','tests'):
        p=physics/directory/'example.F90';p.parent.mkdir(parents=True);p.write_text('module example\nend module\n')
    paths=source_files(tmp_path,'rrfs')
    assert [p.parent.name for p in paths]==['MPAS']
    assert len(source_files(tmp_path))==5
    assert pin_set('rrfs')['model_commit']=='1d6e370e7b43a332074809f170532a216c2212fb'


def test_config_preserves_allocations_types_and_out_entry(tmp_path):
    p=tmp_path/'config.F90'
    p.write_text('''module cfg
 type :: native_cfg
 logical :: option=.true.
 real(kind=8),allocatable :: diagnostic(:,:)
 end type
 end module
''')
    fields=primitive_components(tmp_path,{'path':'config.F90'},'native_cfg','cfg','native','out')
    assert len(fields)==2
    assert fields[1]['type']=='real(kind=8)'
    assert fields[1]['presence_guard']=='allocated(native%diagnostic)'
    assert fields[1]['optional'] and fields[1]['entry_storage']


def test_reference_selector_rejects_cross_family():
    path=ORACLE.parents[1]/'column_oracle_replay.py'
    spec=importlib.util.spec_from_file_location('rrfs_column_replay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.validate_reference({'reference':'rrfs'},'rrfs')=='rrfs'
    assert module.validate_reference({},'mpas')=='mpas'
    with pytest.raises(ValueError,match='reference mismatch'):
        module.validate_reference({'reference':'rrfs'},'mpas')
    with pytest.raises(ValueError,match='reference mismatch'):
        module.validate_reference({},'rrfs')


def test_counter_declarations_precede_array_and_component_assignments():
    # GFL's initialization starts with array assignments before its first CALL.
    for statement in ('q(:,1)=0.', 'cfg(1)%values(:)=0.', 'ptr%field=>target', 'call native()'):
        assert EXECUTABLE.match(statement)
    for declaration in ('real :: q(10)', 'integer,parameter :: k=2', 'use native,only:q'):
        assert not EXECUTABLE.match(declaration)


def test_fixed_derived_arrays_keep_native_element_indices(tmp_path):
    (tmp_path/'config.F90').write_text('''module cfg
 type :: native_cfg
 real(kind=8),allocatable :: diagnostic(:)
 end type
 end module
''')
    fields=primitive_components(tmp_path,{'path':'config.F90'},'native_cfg','cfg','native','inout',['2'])
    assert [f['name'] for f in fields]==['cfg.1.diagnostic','cfg.2.diagnostic']
    assert fields[0]['actual']=='native(1)%diagnostic'
    assert fields[1]['actual']=='native(2)%diagnostic'


def test_statement_functions_remain_before_executable_hooks():
    from types import SimpleNamespace
    routine=SimpleNamespace(variables={
        'galpha':Variable('galpha','real','',[],'inout',False,False),
        'pb':Variable('pb','real','',['10'],'inout',False,False)},
        body=[Statement('galpha(a_in)=a_in*2.',0,1),Statement('pb(:)=0.',1,2),Statement('call native()',2,3)])
    assert first_executable(routine).start==1


def test_initialized_native_arrays_remain_before_entry_hooks():
    from types import SimpleNamespace
    routine=SimpleNamespace(variables={},body=[
        Statement('REAL(RKIND), PARAMETER :: c_old(5)=(/1.373,3.2,0.057,1.05,1.190/)',0,1),
        Statement('REAL(RKIND) :: tc1(imx,jmx,lmx,nmx)',1,2),
        Statement('fudge_fac=.25',2,3)])
    assert first_executable(routine).start==2


def test_standalone_native_access_preserves_private_initializer_body(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from prepare_standalone import expose_native_routines
    source='''module native_module
private
type :: native_type
contains
procedure :: method
end type
contains
subroutine initialize(x)
real :: x
x=0.
end subroutine
end module
'''
    (tmp_path/'native.F90').write_text(source)
    contents={'native.F90':source}
    changes=expose_native_routines(contents,tmp_path,['native_module:initialize'])
    assert contents['native.F90'].replace(changes[0]['insertion'],'')==source
    assert changes[0]['instrumented_lines']==[8,11]
    assert changes[0]['numerical_source_changed'] is False
    assert contents['native.F90'].index(changes[0]['insertion'])>source.index('end type')
    with pytest.raises(ValueError,match='not in the selected source closure'):
        expose_native_routines(contents,tmp_path,['native_module:absent'])


def test_synthetic_capture_uses_fixture_storage_and_zero_argument_calls():
    sys.path.insert(0,str(ORACLE/'replay'))
    from synthetic_capture import generate as synthetic
    site={'issues':[],'arguments':[{'name':'q','actual':'driver_q',
          'type':'real','dims':['ims:ime'],'optional':False,'intent':'inout'}],
          'site':'native_site','scheme':'native','callee':{'module':'native_module',
          'path':'native.F90','line':1,'end_line':10},'state_modules':[]}
    text=synthetic(site,{'dimensions':{'ims':-2,'ime':0},'values':{'q':'1.0_real32'}})
    assert 'driver_q' not in text
    assert 'real(real32) :: q(-2:0)' in text
    assert 'q, oracle_event, lower=lbound(q)' in text
    assert site['arguments'][0]['actual']=='driver_q'
    # The fork's MYJ declarations spell bounds in upper case; Fortran binds
    # those to the same dimensions used by lower-case MPAS declarations.
    site['arguments'][0]['dims']=['IMS:IME']
    assert 'real(real32) :: q(-2:0)' in synthetic(site,{
        'dimensions':{'ims':-2,'ime':0},'values':{'q':'1.0_real32'}})
    from generate_replay import axes
    assert axes(site['arguments'][0])==(1,0)
    assert axes({'dims':['IMS:IME','KMS:KME','JMS:JME']})==(1,2)
    site['arguments']=[]
    assert 'call native()' in synthetic(site,{'dimensions':{},'values':{}})


def test_synthetic_native_logging_initializes_before_logged_scheme(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from synthetic_capture import generate as synthetic
    site={'issues':[], 'arguments':[], 'site':'logged_native', 'scheme':'native',
          'callee':{'module':'native_module','path':'native.F90','line':1,'end_line':10},
          'state_modules':[], 'runtime_modules':['mpas_log']}
    text=synthetic(site,{'dimensions':{},'values':{}})
    assert 'use mpas_log, only: mpas_log_init' in text
    assert text.index('implicit none') < text.index('type(domain_type), pointer')
    assert text.index('replay_domain%dminfo%nprocs=1') < text.index('call mpas_log_init')
    assert text.index('call mpas_log_init') < text.index('call native()')
    from prepare_framework import framework_site
    (tmp_path/'caller.F90').write_text('module caller\ncontains\nsubroutine driver()\nend subroutine\nend module\n')
    original=dict(site,path='caller.F90',driver='driver',runtime_modules=[])
    restored=framework_site(original,tmp_path)
    assert restored['runtime_modules']==['mpas_log']
    assert original['runtime_modules']==[]


def test_synthetic_totals_authenticate_shards_and_preserve_missing_counts(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from run_synthetic import comparison_totals
    spec=importlib.util.spec_from_file_location('synthetic_receipts',ORACLE.parents[1]/'column_oracle_replay.py')
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    receipt=tmp_path/'replay.json'
    arrays={f'field{i}':{'words':i+1,'different_words':i%2,'witness':'x'*180} for i in range(8)}
    harness.write_replay_receipt(receipt,{'captures':[{'capture':'one','site':'native','arrays':arrays}]},limit=900)
    assert json.loads(receipt.read_text())['capture_shards']
    data=harness.read_replay_receipt(receipt)
    assert comparison_totals(data)=={'arrays':8,'compared_words':36,'differing_words':4}
    del data['captures'][0]['arrays']['field0']['different_words']
    assert comparison_totals(data)['differing_words'] is None
    shard=next(tmp_path.glob('replay.captures-*.json'))
    shard.write_text(shard.read_text()+' ')
    with pytest.raises(ValueError,match='shard hash mismatch'):
        harness.read_replay_receipt(receipt)


def test_archived_native_reproduction_refuses_partial_or_unbound_authority(tmp_path):
    sys.path.insert(0,str(ORACLE/'replay'))
    from reproduce_archived_replay import authority
    def put(name,data):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data));return str(path)
    audit=put('audit.json',{'recipe':'gnu-strict','all_ok':True})
    capture=put('capture.json',dict(schema='mpas-oracle-passivity-v1',exit=0,changed_inputs=[],
        process_count=1,comparison_complete=True,all_field_words_equal=True,differing_words=0,
        missing_or_incompatible_samples=0,passivity_byte_equal=False,
        object_audit_sha256=hashlib.sha256(Path(audit).read_bytes()).hexdigest()))
    data=dict(schema='real-native-archived-replay-inputs-v1',build=str(tmp_path/'build'),
        audit=audit,link_manifest=put('link.json',{}),pins=put('pins.json',{}),sites=['native'],
        cases=[dict(id='real',capture=capture,publication=put('publication.json',{}),
                    inventory=put('inventory.json',{}),capture_index=put('index.json',{}),
                    consolidated=put('consolidated.json',{}))])
    put('build/oracle-inventory.json',{})
    paths=[audit,data['link_manifest'],data['pins'],str(tmp_path/'build/oracle-inventory.json'),
           *[v for k,v in data['cases'][0].items() if k!='id']]
    def bind():
        data['dependencies']=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths]
        return Path(put('authority.json',data))
    index=bind()
    assert authority(index)['cases'][0]['capture']==capture
    # Scientific-word equality is explicit; it must not upgrade literal bytes.
    assert json.loads(Path(capture).read_text())['passivity_byte_equal'] is False
    producer=json.loads(Path(capture).read_text());producer['comparison_complete']=False
    put('capture.json',producer)
    with pytest.raises(ValueError,match='dependency differs'):authority(index)
    with pytest.raises(ValueError,match='scientific comparison is incomplete'):authority(bind())
    producer['comparison_complete']=True;put('capture.json',producer);index=bind()
    data['dependencies']=data['dependencies'][1:];index=Path(put('authority.json',data))
    with pytest.raises(ValueError,match='not hash-bound'):authority(index)
    data['producer_audit']=audit
    data['audit']=put('coverage-audit.json',{'recipe':'coverage','all_ok':True})
    paths.append(data['audit']);data['recipe']='coverage'
    index=bind()
    assert authority(index)['producer_audit']==audit
    from collect_archived_coverage import validate
    replay=tmp_path/'replay';counter=replay/'profiles/native.gcda'
    counter.parent.mkdir(parents=True);counter.write_bytes(b'actual counter witness')
    measured=dict(recipe='coverage',comparison_complete=True,process_count=1,differing_words=0,
        input_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),selected_sites=['native'],
        cases=[dict(id='real')],coverage_counters_sha256={'profiles/native.gcda':hashlib.sha256(counter.read_bytes()).hexdigest()})
    put('replay/reproduction.json',measured)
    assert validate(replay,index)[0]['producer_audit']==audit
    portable=tmp_path/'portable';portable.mkdir()
    for row in json.loads(index.read_text())['dependencies']:
        (portable/row['sha256']).write_bytes(Path(row['path']).read_bytes())
    Path(capture).unlink()
    assert validate(replay,index,portable)[0]['producer_audit']==audit
    (portable/hashlib.sha256(Path(audit).read_bytes()).hexdigest()).write_bytes(b'changed audit')
    with pytest.raises(ValueError,match='dependency differs'):validate(replay,index,portable)
    put('capture.json',producer)
    counter.write_bytes(b'changed counter')
    with pytest.raises(ValueError,match='counter changed'):validate(replay,index)
    measured['coverage_counters_sha256']={'../escaped.gcda':'unbound'}
    put('replay/reproduction.json',measured)
    with pytest.raises(ValueError,match='escapes replay'):validate(replay,index)
    measured['comparison_complete']=False;put('replay/reproduction.json',measured)
    with pytest.raises(ValueError,match='complete bound coverage replay required'):validate(replay,index)
    data['recipe']='unrecorded-flags'
    with pytest.raises(ValueError,match='unsupported native replay recipe'):authority(bind())
    data['recipe']='gnu-strict'
    del data['cases'][0]['publication']
    with pytest.raises(ValueError,match='both publication and consolidation'):authority(bind())
    del data['cases'][0]['consolidated']
    with pytest.raises(ValueError,match='absolute archive root'):authority(bind())
    data['cases'][0]['archive_root']=str(tmp_path/'individual-archives')
    assert authority(bind())['cases'][0]['archive_root']==str(tmp_path/'individual-archives')
    data['sites']=[]
    with pytest.raises(ValueError,match='empty or duplicated'):authority(Path(put('authority.json',data)))


def test_synthetic_derived_storage_and_absent_optional_arguments():
    sys.path.insert(0,str(ORACLE/'replay'))
    from synthetic_capture import generate as synthetic
    field={'name':'cfg.weight','actual':'caller_cfg%weight','replay_expr':'cfg%weight',
           'type':'real','dims':[':'],'optional':True,'intent':'inout',
           'presence_guard':'allocated(caller_cfg%weight)'}
    site={'issues':[],'arguments':[{'name':'cfg','actual':'caller_cfg',
          'type':'type(native_cfg)','derived_module':'native_types','dims':[],
          'optional':False,'intent':'inout','components':[field]},
          {'name':'option','actual':'caller_option','type':'integer',
           'dims':[],'optional':True,'intent':'in'}],
          'site':'native_site','scheme':'native','callee':{'module':'native_module',
          'path':'native.F90','line':1,'end_line':10},'state_modules':[]}
    text=synthetic(site,{'dimensions':{},'values':{},'omitted':['option'],
                       'derived_initialize':{'cfg':['  allocate(cfg%weight(-2:0))','  cfg%weight=1.0']}})
    assert 'caller_' not in text
    assert 'use native_types,only:native_cfg' in text
    assert 'allocated(cfg%weight)' in text
    assert "'option.present',.false." in text
    assert 'option=option' not in text
    assert field['actual']=='caller_cfg%weight'


def test_rrfs_publication_authenticates_native_recipe_and_pins(tmp_path):
    config=pin_set('rrfs')
    (tmp_path/'atmosphere_model').write_bytes(b'publication schema fixture')
    (tmp_path/'build-source.diff').write_text('')
    (tmp_path/'flag-audit.json').write_text(json.dumps({'all_ok':True}))
    digest=lambda p:hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()
    manifest={'recipe':'gnu-strict','core':'atmosphere',
              'pin_set':{'fork':config['model_commit'],'submodules':config['submodules']},
              'binary_sha256':digest('atmosphere_model'),'source_diff_sha256':digest('build-source.diff'),
              'flags_sha256':digest('flag-audit.json')}
    manifest['pin_set_hash']=hashlib.sha256(json.dumps(manifest['pin_set'],sort_keys=True).encode()).hexdigest()
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    assert reference_build(tmp_path,'rrfs')[1]==config['model_commit']
    (tmp_path/'atmosphere_model').write_bytes(b'changed')
    with pytest.raises(ValueError,match='binary hash differs'):reference_build(tmp_path,'rrfs')


def test_driver_boundary_condition_is_once_and_pointer_element_is_scalar(tmp_path):
    from boundaries import generate_boundaries
    source=tmp_path/'source';out=tmp_path/'owned'
    physics=source/'src/core_atmosphere/physics';physics.mkdir(parents=True)
    dynamics=source/'src/core_atmosphere/dynamics';dynamics.mkdir()
    (source/'src/core_atmosphere/Registry.xml').write_text('<registry/>')
    (dynamics/'mpas_atm_time_integration.F').write_text('')
    (physics/'mpas_atmphys_driver.F').write_text('''module native_boundary
 contains
 subroutine advance(q)
 real,pointer :: q(:)
 integer :: thread
 thread=1
 if(enabled()) call driver_slice(q(thread),2)
 end subroutine
 end module
''')
    manifest={'pin_set':'rrfs'};generate_boundaries(source,out,manifest)
    text=(out/'src/core_atmosphere/physics/mpas_atmphys_driver.F').read_text()
    assert text.count('enabled()')==1
    assert text.index('if(enabled()) then')<text.index('call oracle_begin')<text.index('call driver_slice')
    assert 'associated(q(thread))' not in text and 'lbound(q(thread))' not in text
    assert manifest['boundaries'][0]['values']==['q(thread)','2']
    assert not manifest['boundaries'][0]['uncaptured_arguments']


def test_nested_scopes_preserve_parent_hidden_inputs_and_final_outputs(tmp_path):
    (tmp_path/'transport.F').write_text(generate())
    (tmp_path/'fixture.F90').write_text('''program fixture
 use mpas_oracle_dump
 implicit none
 integer :: outer,inner,invocation,value,seen
 real,allocatable :: hidden_array(:)
 logical :: exists
 character(len=4096) :: prefix
 call get_command_argument(1,prefix)
 call oracle_begin('outer',outer)
 call oracle_scope_begin('outer',outer)
 value=11
 call oracle_local_begin('parent',invocation)
 call oracle_hidden('local.parent.counter',value,invocation,'pre')
 value=12
 call oracle_hidden('local.parent.counter',value,invocation,'post')
 call oracle_begin('inner',inner)
 call oracle_scope_begin('inner',inner)
 value=21
 call oracle_local_begin('child',invocation)
 call oracle_hidden('local.child.counter',value,invocation,'pre')
 value=22
 call oracle_hidden('local.child.counter',value,invocation,'post')
 call oracle_scope_end()
 value=99
 call oracle_local_begin('child',invocation)
 call oracle_hidden('local.child.counter',value,invocation,'pre')
 value=23
 call oracle_hidden('local.child.counter',value,invocation,'post')
 call oracle_scope_end()
 call oracle_read(trim(prefix)//'outer.s0000000000.e0000000001.r000000.pre.local.child.counter.bin',seen)
 if(seen/=21) error stop 'parent did not preserve first child input'
 call oracle_read(trim(prefix)//'outer.s0000000000.e0000000001.r000000.post.local.child.counter.bin',seen)
 if(seen/=23) error stop 'parent did not preserve final child output'
 call oracle_read(trim(prefix)//'inner.s0000000000.e0000000001.r000000.post.local.child.counter.bin',seen)
 if(seen/=22) error stop 'child output changed by parent'
 call oracle_read(trim(prefix)//'outer.s0000000000.e0000000001.r000000.post.local.parent.counter.bin',seen)
 if(seen/=12) error stop 'parent pending state lost'
 call oracle_replay_begin(trim(prefix)//'outer.s0000000000.e0000000001.r000000','outer',outer)
 if(.not.oracle_replaying()) error stop 'replay mode missing'
 value=-1
 call oracle_local_begin('child',invocation)
 call oracle_hidden('local.child.counter',value,invocation,'pre')
 if(value/=21) error stop 'child hidden input not restored'
 value=25
 call oracle_local_begin('child',invocation)
 call oracle_hidden('local.child.counter',value,invocation,'pre')
 if(value/=25) error stop 'second child entry restored twice'
 call oracle_scope_end()
 if(oracle_replaying()) error stop 'replay mode leaked'
 call oracle_begin('alloc',outer)
 call oracle_scope_begin('alloc',outer)
 allocate(hidden_array(-2:0));hidden_array=[1.,2.,3.]
 call oracle_local_begin('allocation',invocation)
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'pre')
 hidden_array=hidden_array+4.
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'post')
 call oracle_scope_end()
 deallocate(hidden_array)
 call oracle_replay_begin(trim(prefix)//'alloc.s0000000000.e0000000001.r000000','alloc',outer)
 call oracle_local_begin('allocation',invocation)
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'pre')
 if(.not.allocated(hidden_array)) error stop 'allocation not restored'
 if(lbound(hidden_array,1)/=-2.or.any(hidden_array/=[1.,2.,3.])) error stop 'allocation bounds or words changed'
 deallocate(hidden_array)
 call oracle_local_begin('allocation',invocation)
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'pre')
 if(allocated(hidden_array)) error stop 'allocation restored twice'
 call oracle_scope_end()
 call oracle_begin('gone',outer)
 call oracle_scope_begin('gone',outer)
 allocate(hidden_array(1));hidden_array=1.
 call oracle_local_begin('allocation',invocation)
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'pre')
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'post')
 deallocate(hidden_array)
 call oracle_hidden_alloc('local.allocation.array',hidden_array,invocation,'post')
 call oracle_scope_end()
 inquire(file=trim(prefix)//'gone.s0000000000.e0000000001.r000000.post.local.allocation.array.bin',exist=exists)
 if(exists) error stop 'stale deallocated output remained'
end program
''')
    (tmp_path/'Makefile').write_text('all: fixture\nfixture: transport.F fixture.F90\n\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form $^ -o $@\n')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,text=True)
    env=dict(os.environ,MPAS_ORACLE_DIR=str(tmp_path),MPAS_ORACLE_SCHEMES='all')
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path)+'/'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_allocation_hooks_preserve_each_ancestor_invocation(tmp_path):
    """A parent replay must restore both allocations, including fresh bytes."""
    (tmp_path/'transport.F').write_text(generate())
    (tmp_path/'fixture.F90').write_text('''program nested_allocation
 use iso_fortran_env,only:int32,int8
 use mpas_oracle_dump
 implicit none
 integer :: parent,child,i
 real,allocatable :: work(:)
 real :: captured(4,2),replayed(4,2)
 character(len=4096) :: prefix
 call get_command_argument(1,prefix)
 call oracle_begin('alloc_parent',parent)
 call oracle_scope_begin('alloc_parent',parent)
 do i=1,2
  call oracle_begin('alloc_child',child)
  call oracle_scope_begin('alloc_child',child)
  allocate(work(-1:2))
  call oracle_allocation('native.work.L1',work,lower=lbound(work))
  work(-1)=42.0+i
  work(0)=17.0
  captured(:,i)=work
  deallocate(work)
  call oracle_scope_end()
 enddo
 call oracle_scope_end()
 call oracle_replay_begin(trim(prefix)//'alloc_parent.s0000000000.e0000000001.r000000','alloc_parent',parent)
 do i=1,2
  allocate(work(-1:2))
  work=-999.0
  call oracle_allocation('native.work.L1',work,lower=lbound(work))
  work(-1)=42.0+i
  work(0)=17.0
  replayed(:,i)=work
  deallocate(work)
 enddo
 if(any(transfer(captured,[0_int8])/=transfer(replayed,[0_int8]))) error stop 'parent allocation entries not restored'
 call oracle_scope_end()
end program
''')
    (tmp_path/'Makefile').write_text('all: fixture\nfixture: transport.F fixture.F90\n\tgfortran -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -ffree-form $^ -o $@\n')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,text=True)
    env=dict(os.environ,MPAS_ORACLE_DIR=str(tmp_path),MPAS_ORACLE_KEYS='*')
    result=subprocess.run([str(tmp_path/'fixture'),str(tmp_path)+'/'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    parent=sorted(tmp_path.glob('alloc_parent*.pre.allocation.*.bin'))
    child=sorted(tmp_path.glob('alloc_child*.pre.allocation.*.bin'))
    assert len(parent)==len(child)==2
    assert [p.name.rsplit('.',2)[1] for p in parent]==['n0000000001','n0000000002']
    assert all(p.name.endswith('.n0000000001.bin') for p in child)


def test_recorded_native_entry_replay_preserves_context_and_adjacent_words(tmp_path):
    """Actual GNU ABI/Rust ptrace replay; malformed source context fails closed."""
    site_support.tools_or_skip("make", "gfortran", "rustc", what="compiling this native-oracle observer")
    import shutil
    import struct
    shutil.copy2(ORACLE/'replay/native_entry_watch.rs',tmp_path/'observer.rs')
    names=[f'd{i}' for i in range(58)]
    scalar=[n for n in names if n!='d57']
    (tmp_path/'fixture.F90').write_text('''module module_cu_gfl_deep
contains
subroutine cu_gfl_deep_run('''+','.join(names)+''')
 integer :: '''+','.join(scalar)+'''
 integer :: d57(d2:d3)
 d57=d57+100
end subroutine
end module
program fixture
 use module_cu_gfl_deep
 implicit none
 integer :: zero=0,last=3,first=1,imid=1,storage(5)
 character(len=16) :: mode
 call get_command_argument(1,mode)
 storage=[-99,10,0,-3,99]
 if(mode=='alternate') storage=[-99,42,4,5,99]
 call cu_gfl_deep_run('''+','.join({0:'last',2:'first',3:'last',13:'imid',57:'storage(2:4)'}.get(i,'zero') for i in range(58))+''')
 print *,storage
end program
''')
    (tmp_path/'Makefile').write_text('''all: observer fixture
observer: observer.rs
\trustc --edition=2021 -O observer.rs -o $@
fixture: fixture.F90
\tgfortran -O3 -fno-inline -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none $< -o $@
''')
    subprocess.run(_build_command(['make','-j4']),cwd=tmp_path,check=True,capture_output=True,text=True)
    original=subprocess.run([str(tmp_path/'fixture')],check=True,capture_output=True,text=True).stdout
    recorded=tmp_path/'recorded'
    observer=[str(tmp_path/'observer'),str(tmp_path/'fixture')]
    normal=subprocess.run(observer+[str(recorded)],check=True,capture_output=True,text=True,timeout=10)
    assert normal.stdout==original
    event=json.loads((recorded/'rank0.jsonl').read_text())
    assert event['ite']==event['itf']==3 and event['its']==1 and event['imid']==1
    assert (recorded/event['file']).read_bytes()==struct.pack('<3i',10,0,-3)
    (recorded/'entry-context.tsv').write_text('0 0 1 1 3 3 4\n')
    # A wrapper is necessary because the observer executes the supplied binary
    # without extra arguments; mutate the fixture's original initialization.
    source=(tmp_path/'fixture.F90').read_text().replace('storage=[-99,10,0,-3,99]','storage=[-99,42,4,5,99]')
    (tmp_path/'fixture.F90').write_text(source)
    subprocess.run(_build_command(['make','-j4','fixture']),cwd=tmp_path,check=True,capture_output=True,text=True)
    replayed=tmp_path/'replayed'
    result=subprocess.run(observer+[str(replayed),'--replay-entries',str(recorded)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert result.stdout==original  # Both adjacent sentinel words remain intact.
    assert (replayed/event['file']).read_bytes()==(recorded/event['file']).read_bytes()
    assert (replayed/event['file'].replace('.raw','.original.raw')).read_bytes()==struct.pack('<3i',42,4,5)
    comparison=subprocess.run([str(tmp_path/'observer'),'--compare',str(recorded/event['file']),str(replayed/event['file'].replace('.raw','.original.raw'))],check=True,capture_output=True,text=True)
    assert json.loads(comparison.stdout)['different_zero_tests']==1
    (recorded/'entry-context.tsv').write_text('0 0 1 1 3 2 4\n')
    refused=subprocess.run(observer+[str(tmp_path/'mismatched'),'--replay-entries',str(recorded)],capture_output=True,text=True,timeout=10)
    assert refused.returncode!=0 and 'recorded entry context mismatch' in refused.stderr
    assert not list((tmp_path/'mismatched').glob('*.raw'))


def test_external_source_abi_observes_complete_native_types_aliases_and_locals(tmp_path):
    """Native calls remain untouched; Rust packets match all explicit ABI shapes."""
    site_support.tools_or_skip("make", "gfortran", "rustc", what="compiling this native-oracle observer")
    import shutil
    from generate import inventory
    from external_abi import emit
    root=tmp_path/'source';root.mkdir()
    native=root/'src/core_atmosphere/physics/physics_wrf/physics.F90'
    native.parent.mkdir(parents=True)
    native.write_text('''module physics
contains
subroutine hidden(variant,lo,hi,payload)
 integer,intent(in) :: variant,lo,hi
 integer,intent(inout) :: payload(lo:hi)
 payload=payload+variant
end subroutine
subroutine run(itimestep,lo,hi,values,alias,text,mask,r8,cx)
 integer,intent(in) :: itimestep,lo,hi
 real,intent(inout) :: values(lo:hi)
 real,intent(inout),optional :: alias(lo:hi)
 character(*),intent(in) :: text
 logical,intent(in) :: mask
 real(8),intent(inout) :: r8
 complex(8),intent(inout) :: cx
 integer :: flag,hidden_a(lo:hi),hidden_b(lo:hi)
 hidden_a=7; hidden_b=9; flag=1
 if(flag==1)then
 call hidden(flag,lo,hi,hidden_a)
 endif
 call hidden(0,lo,hi,hidden_b)
 values=values+real(hidden_a+hidden_b)
 if(present(alias)) alias=values+3
 if(mask) r8=r8*2
 cx=cx+(1.,2.)
 print *,text,values,r8,cx
end subroutine
subroutine driver_external(itimestep,lo,hi,values,text,mask,r8,cx)
 integer,intent(in) :: itimestep,lo,hi
 real,intent(inout) :: values(lo:hi)
 character(*),intent(in) :: text
 logical,intent(in) :: mask
 real(8),intent(inout) :: r8
 complex(8),intent(inout) :: cx
 call run(itimestep,lo,hi,values,values,text,mask,r8,cx)
 call run(itimestep+1,lo,hi,values,text=text,mask=mask,r8=r8,cx=cx)
end subroutine
end module
program fixture
 use physics
 real :: values(-2:0)=[1.,2.,3.]
 real(8) :: r8=1.5
 complex(8) :: cx=(2.,3.)
 call driver_external(4,-2,0,values,'native chars',.true.,r8,cx)
end program
''')
    manifest=inventory(root,authenticate=False)
    generated=tmp_path/'generated'
    emit(root,generated,manifest,['run'],['hidden:payload'])
    build=generated/'run'
    shutil.copy2(native,build/'physics.F90')
    shutil.copy2(ORACLE/'oracle_words.rs',build/'oracle_words.rs')
    with (build/'Makefile').open('a') as f:
        f.write('fixture: physics.F90\n\tgfortran -O3 -fno-frontend-optimize -fno-inline -fno-inline-functions -fno-inline-small-functions -fno-inline-functions-called-once -fno-ipa-cp -fno-ipa-sra -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    subprocess.run(_build_command(['make','-j4','observer','fixture','oracle_words']),cwd=build,check=True,capture_output=True,text=True)
    ordinary=subprocess.run([str(build/'fixture')],check=True,capture_output=True,text=True).stdout
    packets=tmp_path/'packets'
    observed=subprocess.run([str(build/'observer'),str(build/'fixture'),str(packets)],check=True,capture_output=True,text=True,timeout=10)
    assert observed.stdout==ordinary
    calls=[json.loads(l) for l in (packets/'rank0-calls.jsonl').read_text().splitlines()]
    assert len(calls)==4 and calls[0]['argument_aliases']=={'alias':'values'}
    assert calls[2]['argument_aliases']=={}
    absent=packets/(calls[2]['prefix']+'.pre.alias.present.bin')
    assert subprocess.check_output([str(build/'oracle_words'),'present',str(absent)],text=True).strip()=='false'
    assert not calls[0]['global_cell_ids_known'] and not calls[0]['hidden_state_complete']
    prefix=calls[0]['prefix'];words=str(build/'oracle_words')
    for name,code,width,shape in [('values',1,4,[3]),('alias',1,4,[3]),('text',5,1,[]),('mask',4,4,[]),('r8',2,8,[]),('cx',8,16,[])]:
        packet=packets/(prefix+'.pre.'+name+'.bin')
        header=json.loads(subprocess.check_output([words,'header',str(packet)],text=True))
        assert header['code']==code and header['width']==width and header['shape']==shape
    for name in ('hidden_a','hidden_b'):
        assert (packets/(prefix+'.pre.local.physics.'+name+'.bin')).is_file()
    from replay.replay_external_calls import compare_outputs
    arguments=json.loads((build/'spec.json').read_text())['arguments']
    for call in calls[::2]:
        captured=tmp_path/('captured-'+str(call['call']));captured.mkdir()
        replayed=tmp_path/('replayed-'+str(call['call']));replayed.mkdir()
        for a in arguments:
            if a['intent']=='in':continue
            labels=[a['name']+'.present',a['name']] if a['optional'] else [a['name']]
            for label in labels:
                name=call['prefix']+'.post.'+label+'.bin'
                if (packets/name).exists():
                    shutil.copy2(packets/name,captured/name)
                    shutil.copy2(packets/name,replayed/('replay.post.'+label+'.bin'))
        arrays,complete=compare_outputs(arguments,captured,call['prefix'],replayed,words)
        assert complete and all(a['different_words']==0 for a in arrays.values())
        if call['call']==0:
            # A present optional output missing on both sides is incomplete.
            (captured/(call['prefix']+'.post.alias.bin')).unlink()
            (replayed/'replay.post.alias.bin').unlink()
            arrays,complete=compare_outputs(arguments,captured,call['prefix'],replayed,words)
            assert not complete and arrays['alias']['status']=='missing'
        else:
            # Missing presence records cannot certify an absent argument.
            (captured/(call['prefix']+'.post.alias.present.bin')).unlink()
            (replayed/'replay.post.alias.present.bin').unlink()
            arrays,complete=compare_outputs(arguments,captured,call['prefix'],replayed,words)
            assert not complete and arrays['alias.present']['status']=='missing'
    comparison=json.loads(subprocess.check_output([words,'compare',str(packets/(prefix+'.pre.values.bin')),str(packets/(prefix+'.pre.alias.bin'))],text=True))
    assert comparison['different_words']==0 and comparison['words']==3
    # Exercise the original external transport's tagged metadata and scalar
    # character encoding through an actual Rust producer, preserving payloads.
    legacy=(build/'native_argument_watch.rs').read_text().replace(
        'for value in [SYMBOL,phase,name]',
        'for value in [name,phase,"pristine-external-wrapper-unknown-global-IDs"]')
    legacy=legacy.replace('h[3]=data.len() as i32;h[1]=1;', 'h[3]=1;h[1]=data.len() as i32;')
    (build/'legacy.rs').write_text(legacy)
    with (build/'Makefile').open('a') as f:
        f.write('legacy: legacy.rs native_entry_watch.rs argument_spec.rs\n\trustc --edition=2021 -O $< -o $@\n')
    subprocess.run(_build_command(['make','-j4','legacy']),cwd=build,check=True,capture_output=True,text=True)
    legacy_packets=tmp_path/'legacy-packets'
    result=subprocess.run([str(build/'legacy'),str(build/'fixture'),str(legacy_packets)],check=True,capture_output=True,text=True)
    assert result.stdout==ordinary
    for name in ('values','text','cx','local.physics.hidden_a'):
        packet_name=prefix+'.pre.'+name+'.bin'
        normalized=tmp_path/('normalized-'+name+'.bin')
        result=json.loads(subprocess.check_output([words,'normalize-external',str(legacy_packets/packet_name),str(normalized)],text=True))
        assert result['payload_unchanged'] and result['ids_unchanged']
        comparison=json.loads(subprocess.check_output([words,'compare',str(packets/packet_name),str(normalized)],text=True))
        assert comparison['different_words']==0 and comparison['bitwise_equal']


def test_external_abi_rejects_unknown_descriptors_and_nonscalar_bounds(tmp_path):
    from external_abi import variable_layout,rust_expression
    from fortran_source import Routine,Variable
    routine=Routine('run','physics','physics.F90',['n','x'],0,1,[],{
        'n':Variable('n','integer','','[]','in',False,False),
        'x':Variable('x','real','','[:]','inout',False,False)})
    routine.variables['n'].dims=[];routine.variables['x'].dims=[':']
    with pytest.raises(ValueError,match='explicit-shape'):
        variable_layout(routine.variables['x'],routine)
    routine.variables['x'].dims=['1:n'];routine.variables['x'].attrs=',pointer'
    with pytest.raises(ValueError,match='dummy storage'):
        variable_layout(routine.variables['x'],routine)
    with pytest.raises(ValueError,match='integer scalar'):
        rust_expression('x+1',routine)


def test_external_gnu_descriptor_matches_native_strided_assumed_shape_arrays(tmp_path):
    """Actual GNU ABI, reversed/strided ranks, dummy bounds and optional absence."""
    site_support.tools_or_skip("make", "gfortran", "rustc", what="compiling this native-oracle observer")
    import shutil
    from generate import inventory
    from external_abi import emit
    root=tmp_path/'source';native=root/'src/core_atmosphere/physics/physics_wrf/descriptor.F90'
    native.parent.mkdir(parents=True)
    native.write_text('''module descriptor_fixture
contains
subroutine run(itimestep,values,double_values,mask,labels,optional_array)
 use mpas_oracle_dump
 integer,intent(in) :: itimestep
 real,intent(inout) :: values(-1:,:)
 real(8),intent(inout) :: double_values(:)
 logical,intent(inout) :: mask(:)
 character(len=7),intent(inout) :: labels(-2:)
 real,intent(inout),optional :: optional_array(:,:,:)
 call oracle_set_output('native'//achar(48+itimestep))
 call oracle_write('run','pre','values',values,1,lower=lbound(values),cell_axis=0,level_axis=0)
 call oracle_write('run','pre','double_values',double_values,1,lower=lbound(double_values),cell_axis=0,level_axis=0)
 call oracle_write('run','pre','mask',mask,1,lower=lbound(mask),cell_axis=0,level_axis=0)
 call oracle_write('run','pre','labels',labels,1,lower=lbound(labels),cell_axis=0,level_axis=0)
 if(present(optional_array)) call oracle_write('run','pre','optional_array',optional_array,1,lower=lbound(optional_array),cell_axis=0,level_axis=0)
 values=values+7.;double_values=double_values*2.;mask=.not.mask
 labels='changed'
 if(present(optional_array)) optional_array=optional_array+5.
 call oracle_write('run','post','values',values,1,lower=lbound(values),cell_axis=0,level_axis=0)
 call oracle_write('run','post','double_values',double_values,1,lower=lbound(double_values),cell_axis=0,level_axis=0)
 call oracle_write('run','post','mask',mask,1,lower=lbound(mask),cell_axis=0,level_axis=0)
 call oracle_write('run','post','labels',labels,1,lower=lbound(labels),cell_axis=0,level_axis=0)
 if(present(optional_array)) call oracle_write('run','post','optional_array',optional_array,1,lower=lbound(optional_array),cell_axis=0,level_axis=0)
 print *,sum(values),sum(double_values),count(mask)
end subroutine
subroutine driver_descriptor(values,double_values,mask,labels,optional_array)
 real :: values(-2:6,-3:3)
 real(8) :: double_values(-5:5)
 logical :: mask(-4:4)
 character(len=7) :: labels(9)
 real :: optional_array(7,6,5)
 call run(1,values(6:-2:-2,3:-3:-2),double_values(5:-5:-2),mask(-4:4:2),labels(9:1:-2),optional_array(1:7:2,6:1:-2,1:5:2))
 call run(2,values(-2:6:2,-3:3:2),double_values(-5:5:2),mask(4:-4:-2),labels(1:9:2))
end subroutine
end module
program fixture
 use descriptor_fixture
 real :: values(-2:6,-3:3),optional_array(7,6,5)
 real(8) :: double_values(-5:5)
 logical :: mask(-4:4)
 character(len=7) :: labels(9)
 values=reshape([(real(i),i=1,63)],shape(values));optional_array=2.
 double_values=[(real(i,8),i=-5,5)];mask=.true.
 labels=[('entry'//achar(48+i)//' ',i=1,9)]
 call driver_descriptor(values,double_values,mask,labels,optional_array)
end program
''')
    manifest=inventory(root,authenticate=False);generated=tmp_path/'generated'
    emit(root,generated,manifest,['run']);build=generated/'run'
    shutil.copy2(native,build/'descriptor.F90')
    (build/'transport.F').write_text(generate())
    shutil.copy2(ORACLE/'oracle_words.rs',build/'oracle_words.rs')
    with (build/'Makefile').open('a') as f:
        f.write('fixture: transport.F descriptor.F90\n\tgfortran -O3 -fno-frontend-optimize -fno-inline -fno-ipa-cp -fno-ipa-sra -ffree-line-length-none -ffree-form $^ -o $@\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    subprocess.run(_build_command(['make','-j4','observer','fixture','oracle_words']),cwd=build,check=True,capture_output=True,text=True)
    ordinary_dir=tmp_path/'ordinary';observed_dir=tmp_path/'observed'
    ordinary_dir.mkdir();observed_dir.mkdir()
    ordinary=subprocess.run([str(build/'fixture')],cwd=ordinary_dir,check=True,capture_output=True,text=True).stdout
    packets=tmp_path/'packets'
    observed=subprocess.run([str(build/'observer'),str(build/'fixture'),str(packets),'--'],cwd=observed_dir,check=True,capture_output=True,text=True,timeout=20)
    assert observed.stdout==ordinary
    calls=[json.loads(l) for l in (packets/'rank0-calls.jsonl').read_text().splitlines()]
    assert len(calls)==4
    assert json.loads((build/'spec.json').read_text())['descriptor_arguments']==['values','double_values','mask','labels','optional_array']
    words=str(build/'oracle_words')
    for call in calls:
        for name in ('values','double_values','mask','labels','optional_array'):
            if name=='optional_array' and call['call']==1:continue
            reference=observed_dir/(f"native{call['step']}.{call['phase']}.{name}.bin")
            packet=packets/(call['prefix']+'.'+call['phase']+'.'+name+'.bin')
            comparison=json.loads(subprocess.check_output([words,'compare',str(reference),str(packet)],text=True))
            assert comparison['bitwise_equal'],(name,comparison)
    absent=packets/(calls[2]['prefix']+'.pre.optional_array.present.bin')
    assert subprocess.check_output([words,'present',str(absent)],text=True).strip()=='false'


def test_nested_standalone_authority_refuses_mutation_and_forecast_claims(tmp_path):
    """Nested call authority is hash-bound and cannot certify forecast passivity."""
    import copy,hashlib
    from replay.replay_external_calls import standalone_captures
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    packets=tmp_path/'packets';packets.mkdir()
    entry=dict(call=0,step=1,phase='pre',prefix='nested',argument_aliases={},
               hidden_state_complete=False,global_cell_ids_known=False)
    (packets/'rank0-calls.jsonl').write_text(json.dumps(entry)+'\n'+json.dumps(dict(entry,phase='post'))+'\n')
    native=tmp_path/'native-source';native.write_text('immutable native source authority')
    row=dict(parent_row=0,comparison_complete=True,differing_words=0,changed_inputs=[],
        dump=str(packets),capture_sha256={p.name:digest(p) for p in packets.iterdir()},child_calls=1,
        parent_capture_authority={'original_forecast_passive':True},parent_comparison_sha256='bound parent comparison',
        forecast_proven=False,observer_forecast_passive=False)
    manifest=tmp_path/'call.json';manifest.write_text(json.dumps(row))
    spec=dict(pristine_source_sha256={'native':digest(native)})
    index=dict(schema='native-nested-standalone-capture-v1',comparison_complete=True,differing_words=0,
        forecast_proven=False,observer_forecast_passive=False,child_pristine_source_sha256=spec['pristine_source_sha256'],
        dependencies={str(native):digest(native)},rows=[dict(manifest=str(manifest),sha256=digest(manifest))])
    authority=tmp_path/'index.json'
    authority.write_text(json.dumps(index));captured=list(standalone_captures(authority,spec))
    assert len(captured)==1 and captured[0][-1]['observer_forecast_passive'] is False
    for field in ('forecast_proven','observer_forecast_passive'):
        changed=copy.deepcopy(index);changed[field]=True;authority.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='explicitly outside forecast proof'):
            list(standalone_captures(authority,spec))
    authority.write_text(json.dumps(index));manifest.write_text(json.dumps(dict(row,differing_words=1)))
    with pytest.raises(ValueError,match='manifest changed'):
        list(standalone_captures(authority,spec))
    manifest.write_text(json.dumps(row));(packets/'extra.bin').write_bytes(b'opaque changed packet inventory')
    with pytest.raises(ValueError,match='packet inventory changed'):
        list(standalone_captures(authority,spec))
    (packets/'extra.bin').unlink();native.write_text('changed native authority')
    with pytest.raises(ValueError,match='dependency changed'):
        list(standalone_captures(authority,spec))


def test_private_native_module_replay_preserves_partial_out_entry(tmp_path):
    """A real compiler interface exports the whole copied module for replay only."""
    site_support.tools_or_skip("make", "gfortran", "rustc", what="compiling this native-oracle observer")
    import shutil
    from generate import inventory
    from external_abi import emit,prepare_native_replay
    from replay.replay_external_calls import compare_outputs
    root=tmp_path/'source'
    native=root/'src/core_atmosphere/physics/physics_wrf/private_physics.F90'
    native.parent.mkdir(parents=True)
    original="""module private_physics
 private
 public :: driver_private
contains
 subroutine diagnose(lo,hi,values,partial)
 integer,intent(in) :: lo,hi
 real,intent(inout) :: values(lo:hi)
 real,intent(out) :: partial(lo:hi)
 values=values*2
 partial(hi)=values(lo)
 end subroutine
 subroutine driver_private(lo,hi,values,partial)
 integer,intent(in) :: lo,hi
 real,intent(inout) :: values(lo:hi),partial(lo:hi)
 call diagnose(lo,hi,values,partial)
 end subroutine
end module
"""
    native.write_text(original)
    manifest=inventory(root,authenticate=False)
    emit(root,tmp_path/'observer',manifest,['diagnose'])
    observer=tmp_path/'observer/diagnose'
    spec=json.loads((observer/'spec.json').read_text())
    replay=tmp_path/'replay'
    replay_spec=prepare_native_replay(root,spec,manifest,replay,recipes=('gnu-strict',))
    assert replay_spec['standalone_visibility_export'] is True
    assert native.read_text()==original
    generated=replay/'gnu-strict'/native.relative_to(root)
    text=generated.read_text()
    assert 'public :: diagnose' in text
    assert text.replace(' use mpas_oracle_dump,only:oracle_hidden\n','').split(' subroutine diagnose',1)[1].split(' end subroutine',1)[0]==original.split(' subroutine diagnose',1)[1].split(' end subroutine',1)[0]
    shutil.copy2(native,observer/'native.F90')
    shutil.copy2(ORACLE/'oracle_words.rs',observer/'oracle_words.rs')
    (observer/'fixture.F90').write_text("""program fixture
 use private_physics,only:driver_private
 real :: values(-1:1)=[1.,-0.,3.],partial(-1:1)=[-7.,8.,9.]
 call driver_private(-1,1,values,partial)
 print *,values,partial
end program
""")
    with (observer/'Makefile').open('a') as f:
        f.write('fixture: native.F90 fixture.F90\n\tgfortran -O0 $^ -o $@\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    subprocess.run(_build_command(['make','-j4','observer','fixture','oracle_words']),cwd=observer,check=True,capture_output=True,text=True)
    subprocess.run(_build_command(['make','-j4']),cwd=replay,check=True,capture_output=True,text=True)
    ordinary=subprocess.check_output([str(observer/'fixture')],text=True)
    packets=tmp_path/'packets'
    observed=subprocess.check_output([str(observer/'observer'),str(observer/'fixture'),str(packets)],text=True)
    assert observed==ordinary
    calls=[json.loads(line) for line in (packets/'rank0-calls.jsonl').read_text().splitlines()]
    assert len(calls)==2
    output=tmp_path/'outputs';output.mkdir()
    prefix=calls[0]['prefix']
    subprocess.run([str(replay/'gnu-strict/replay'),str(packets/prefix),str(output/'replay')],check=True,capture_output=True,text=True)
    arrays,complete=compare_outputs(replay_spec['site']['arguments'],packets,prefix,output,observer/'oracle_words')
    assert complete and set(arrays)=={'values','partial'}
    assert all(array['bitwise_equal'] for array in arrays.values())


def _legacy_reference_binding_fixture():
    inventory=dict(schema='inventory',sites=[{'site':'complete_driver'}],source_sha256={'src/driver.F':'1'*64},
                   argument_complete=True,hidden_state_complete=False,reference='rrfs',model_commit='fork',
                   submodule_commits={'src/physics':'physics-pin'},pin_set_sha256='2'*64,
                   strict_reference_manifest_sha256='3'*64,strict_reference_binary_sha256='4'*64,
                   generated_sha256={'src/driver.F':'5'*64})
    manifest=dict(schema='mpas-column-dump-v1',recipe='gnu-strict',inventory_sha256='6'*64,
                  sites=inventory['sites'],source_sha256=inventory['source_sha256'],
                  argument_complete=True,hidden_state_complete=False,pins='actual recorded pins\n',
                  pin_set_sha256=hashlib.sha256(b'actual recorded pins\n').hexdigest(),
                  build_provenance={k:inventory[k] for k in ('strict_reference_manifest_sha256',
                                                         'strict_reference_binary_sha256','generated_sha256')})
    selected={k:inventory[k] for k in ('reference','model_commit','submodule_commits','pin_set_sha256')}
    return manifest,inventory,selected


def test_legacy_reference_binding_preserves_inventory_and_missing_state_scope():
    from rebind_dump import bind_inventory
    manifest,inventory,selected=_legacy_reference_binding_fixture()
    original=json.dumps(manifest,sort_keys=True)
    bound=bind_inventory(manifest,inventory,'6'*64,{'binary_sha256':'4'*64},'3'*64,selected,manifest['pins'])
    assert bound==inventory and bound is not inventory
    assert not bound['hidden_state_complete']
    assert json.dumps(manifest,sort_keys=True)==original and 'reference' not in manifest


@pytest.mark.parametrize('mutation', ['inventory','source','interface','build','referee','explicit-reference','pins','pin-set','optimized'])
def test_legacy_reference_binding_rejects_unearned_labels(mutation):
    from rebind_dump import bind_inventory
    manifest,inventory,selected=_legacy_reference_binding_fixture()
    if mutation=='inventory':manifest['inventory_sha256']='0'*64
    elif mutation=='source':manifest['source_sha256']={'src/driver.F':'0'*64}
    elif mutation=='interface':manifest['sites']=[{'site':'other_driver'}]
    elif mutation=='build':manifest['build_provenance']['generated_sha256']={'src/driver.F':'0'*64}
    elif mutation=='referee':inventory['strict_reference_binary_sha256']='0'*64
    elif mutation=='explicit-reference':manifest['reference']='mpas'
    elif mutation=='pins':manifest['pins']='different pins\n'
    elif mutation=='pin-set':selected=dict(selected,pin_set_sha256='0'*64)
    elif mutation=='optimized':manifest['recipe']='gnu-o3'
    with pytest.raises(ValueError):
        bind_inventory(manifest,inventory,'6'*64,{'binary_sha256':'4'*64},'3'*64,selected,'actual recorded pins\n')


@pytest.mark.parametrize('mutation', ['partial', 'missing-post', 'extra-call',
    'wrong-reference', 'changed-receipt', 'changed-counter', 'escaped-counter',
    'wrong-source', 'wrong-pin'])
def test_column_coverage_requires_complete_real_comparison_inventory(tmp_path, monkeypatch, mutation):
    sys.path.insert(0, str(ORACLE/'replay'))
    import collect_column_coverage as collector
    def put(name, value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return p
    index=put('index.json', {})
    audit=put('audit.json', {})
    monkeypatch.setattr(collector, 'authority', lambda *_:dict(recipe='coverage',audit=str(audit),
        build=str(tmp_path/'build')))
    bank=dict(reference='rrfs',dump_root_sha256='bank',pin_set_sha256='pin',
        source_sha256={'native.F':'source'},captures={'call':dict(site='driver')},
        files={'call.post.output.bin':'payload','call.post.state.save.bin':'saved'})
    manifest=put('manifest.json',bank)
    native=dict(reference='rrfs',pin_set_sha256='pin',source_sha256=bank['source_sha256'])
    put('build/oracle-inventory.json',native)
    values={name:dict(words=3,different_words=0,bitwise_equal=True)
            for name in ('output','state.save')}
    receipt=dict(schema='mpas-column-replay-v1',comparison_complete=True,
        all_outputs_bitwise_equal=True,reference='rrfs',dump_root_sha256='bank',
        pin_set_sha256='pin',source_sha256=bank['source_sha256'],replay_recipe='coverage',
        replay_process_count=1,backend='native-replay',resolved_scheme='driver',
        captures=[dict(capture='call',arrays=values)])
    comparison=put('replay/call.json',receipt)
    counter=tmp_path/'replay/profiles/native.gcda';counter.parent.mkdir();counter.write_bytes(b'counter')
    row=dict(capture='call',exit=0,compared_words=6,differing_words=0,
             receipt_sha256=collector.sha(comparison))
    summary=dict(recipe='coverage',comparison_complete=True,process_count=1,differing_words=0,
        input_index_sha256=collector.sha(index),manifest_sha256=collector.sha(manifest),
        reference='rrfs',calls=1,audit_sha256=collector.sha(audit),cases=[row],
        compared_words=6,arrays=2,coverage_counters_sha256={'profiles/native.gcda':collector.sha(counter)})
    put('replay/summary.json',summary)
    assert collector.validate_column(tmp_path/'replay',index,manifest)[0]['sites']==['driver']
    if mutation=='partial':summary['comparison_complete']=False
    elif mutation=='missing-post':
        receipt['captures'][0]['arrays'].pop('state.save');put('replay/call.json',receipt)
        row['receipt_sha256']=collector.sha(comparison)
    elif mutation=='extra-call':summary['cases'].append(dict(row))
    elif mutation=='wrong-reference':
        receipt['reference']='mpas';put('replay/call.json',receipt);row['receipt_sha256']=collector.sha(comparison)
    elif mutation=='changed-receipt':comparison.write_text('changed comparison')
    elif mutation=='changed-counter':counter.write_bytes(b'changed counter')
    elif mutation in ('wrong-source','wrong-pin'):
        if mutation=='wrong-source':native['source_sha256']={'native.F':'different'}
        else:native['pin_set_sha256']='different'
        put('build/oracle-inventory.json',native)
    else:summary['coverage_counters_sha256']={'../escaped.gcda':'unbound'}
    put('replay/summary.json',summary)
    with pytest.raises(ValueError):collector.validate_column(tmp_path/'replay',index,manifest)
