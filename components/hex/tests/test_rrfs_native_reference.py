"""Compiler receipts must prove the exact object and all pinned NOAA groups."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import pytest

# The 2026-10-14 07:45Z source-authority ruling record carries private context, so it is kept in the
# reference store (each init derivative publication's DREW-RULING.json), not in this tree; these tests
# bind its exact bytes and run where RRFS_SOURCE_AUTHORITY_RULING names it (the reference hosts).
_RULING_TESTS = {
    "test_restart_only_forecast_preserves_approved_initializer_authority",
    "test_pins_separate_approved_authorities_from_pristine",
    "test_content_addressed_transport_preserves_case_and_restart_authority",
    "test_canonical_catalogue_binds_complete_cases_and_companions",
    "test_approved_source_authority_rejects_changed_binding",
    "test_approved_initializer_requires_exact_named_binary",
    "test_approved_case_input_authority_rejects_lost_or_changed_provenance",
}


@pytest.fixture(autouse=True)
def _ruling_record(request):
    needs = (request.node.originalname in _RULING_TESTS
             or request.node.name == "test_published_field_recomparison_rejects_untrusted_archives[lost-derivative-label]")
    ruling = os.environ.get("RRFS_SOURCE_AUTHORITY_RULING")
    if needs and not (ruling and Path(ruling).is_file()):
        pytest.skip("source-authority ruling record not supplied (RRFS_SOURCE_AUTHORITY_RULING)")

ROOT = Path(__file__).parents[1] / 'tools/native_physics_reference/rrfs'
spec = importlib.util.spec_from_file_location('rrfs_flags', ROOT / 'audit_flags.py')
flags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flags)


@pytest.mark.parametrize('field', ['patch_sha256', 'binary_sha256', 'ruling_sha256', 'label', 'ruling'])
def test_approved_source_authority_rejects_changed_binding(monkeypatch, field):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_ruled_authority', ROOT/'ruled_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    authority = module.authority('init', module.INIT_BINARY)
    authority[field] = {} if field=='ruling' else 'different'
    with pytest.raises(ValueError, match='binding differs|binary differs'):
        module.validate_authority(authority)


def test_approved_initializer_requires_exact_named_binary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_ruled_init', ROOT/'ruled_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='binary differs'):
        module.authority('init', 'another binary')


@pytest.mark.parametrize('defect', ['case-number','source-binding','input-bytes','input-declaration','legacy-diagnostic'])
def test_approved_case_input_authority_rejects_lost_or_changed_provenance(tmp_path,monkeypatch,defect):
    import hashlib
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_ruled_case',ROOT/'ruled_reference.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    approved=module.authority('init',module.INIT_BINARY)
    output=tmp_path/'init.nc';output.write_bytes(b'actual archived output')
    manifest=dict(schema='mpas.rrfs-native-input/v1',binary_sha256=module.INIT_BINARY,
        source_correction=approved,source_pins=dict(diagnostic=None,source_correction=approved),
        row=dict(variables=dict(init_case=7)),outputs={'init.nc':hashlib.sha256(output.read_bytes()).hexdigest()})
    case={'files':{'init.nc':str(output)}}
    if defect=='case-number':manifest['row']['variables']['init_case']=3
    elif defect=='source-binding':manifest['source_pins']['source_correction']={}
    elif defect=='input-bytes':output.write_bytes(b'changed output')
    elif defect=='input-declaration':case['init_derivative_authorities']=[]
    else:manifest['source_pins']['diagnostic']='counterfactual-optional-aerosol-nullify'
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):module.input_authorities(case)


@pytest.mark.parametrize('defect', [None, 'bytes', 'incomplete', 'label', 'ruling', 'kind', 'declaration', 'missing-binding'])
def test_restart_only_forecast_preserves_approved_initializer_authority(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_restart_authority', ROOT/'ruled_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    approved = module.authority('init', module.INIT_BINARY)
    output = tmp_path/'restart.nc'; output.write_bytes(b'native completed restart')
    receipt = dict(schema='mpas.rrfs-native-forecast/v1', completed=True, exit_status=0,
        outputs={output.name: module.sha(output)}, init_derivative_authorities=[approved],
        reference_labels=[approved['label']])
    case = dict(files={'restart.nc': str(output)})
    if defect == 'bytes': output.write_bytes(b'changed restart')
    elif defect == 'incomplete': receipt['completed'] = False
    elif defect == 'label': receipt['reference_labels'] = []
    elif defect == 'ruling': approved['ruling_sha256'] = 'changed'
    elif defect == 'kind': receipt['init_derivative_authorities'] = [module.authority('gfl', 'binary')]
    elif defect == 'declaration': case['init_derivative_authorities'] = []
    elif defect == 'missing-binding': receipt['init_derivative_authorities'] = []
    (tmp_path/'MANIFEST.json').write_text(json.dumps(receipt))
    if defect:
        with pytest.raises(ValueError): module.input_authorities(case)
    else:
        assert module.input_authorities(case) == [approved]


@pytest.mark.parametrize('defect', [None, 'bytes', 'ruling', 'missing-binding', 'kind'])
def test_content_addressed_transport_preserves_case_and_restart_authority(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    from rrfs import ruled_reference as ruled
    spec = importlib.util.spec_from_file_location('rrfs_derivative_transport', ROOT/'transport_inputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    approved = ruled.authority('init', ruled.INIT_BINARY)
    original = tmp_path/'native'; original.mkdir()
    output = original/'restart.nc'; output.write_bytes(b'actual original restart bytes')
    (original/'MANIFEST.json').write_text(json.dumps(dict(schema='mpas.rrfs-native-forecast/v1',
        completed=True, exit_status=0, outputs={output.name: module.sha(output)},
        init_derivative_authorities=[approved], reference_labels=[approved['label']])))
    packed = tmp_path/'packed'
    receipt = module.pack({str(output): module.sha(output)}, packed, chunk_bytes=8)
    assert receipt['files'][str(output)]['init_derivative_authorities'] == [approved]
    restored = tmp_path/'restored'
    mapping = module.unpack(packed, restored)
    case = dict(files={'restart.nc': mapping[str(output)]}, init_derivative_authorities=[approved])
    assert ruled.input_authorities(case) == [approved]
    if defect == 'bytes': Path(mapping[str(output)]).write_bytes(b'changed')
    elif defect in ('ruling', 'missing-binding', 'kind'):
        p = restored/'INPUT-AUTHORITIES.json'; data = json.loads(p.read_text())
        row = data['files'][module.sha(output)]
        if defect == 'ruling': row['init_derivative_authorities'][0]['ruling_sha256'] = 'changed'
        elif defect == 'missing-binding': del data['files'][module.sha(output)]
        else: row['init_derivative_authorities'] = [ruled.authority('gfl', 'binary')]
        p.write_text(json.dumps(data))
    if defect:
        with pytest.raises(ValueError): ruled.input_authorities(case)
    else:
        # A later unlabelled alias of the same bytes must not erase the ruling.
        alias = tmp_path/'pristine-alias'; alias.write_bytes(output.read_bytes())
        module.pack({str(alias): module.sha(alias)}, tmp_path/'alias-packed', chunk_bytes=8)
        module.unpack(tmp_path/'alias-packed', restored)
        assert ruled.input_authorities(case) == [approved]


@pytest.mark.parametrize('defect', ['publication-bytes', 'incomplete', 'banner', 'init-bytes', 'field-list'])
def test_rrfs_verification_contract_rejects_unbound_native_evidence_before_describing(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_verification_contract', ROOT/'verification_manifest.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path/'forecast'; source.mkdir()
    original = tmp_path/'case'; original.mkdir()
    initial = original/'init.nc'; initial.write_bytes(b'unchanged native input')
    (source/'namelist.atmosphere').write_text('&config\n/\n')
    (source/'streams.atmosphere').write_text('<streams><stream name="output" type="output" filename_template="out.nc"><file name="field-list"/></stream></streams>')
    (source/'field-list').write_text('theta\n')
    (source/'out.nc').write_bytes(b'original native output')
    (source/'log.atmosphere.0000.out').write_text('Finished running the atmosphere core')
    ref = lambda p: dict(path=str(p), sha256=module.sha(p))
    inputs = {k:ref(source/(k+'.atmosphere')) for k in ('namelist','streams')}
    inputs['init'] = ref(initial)
    native = dict(schema='mpas.rrfs-native-forecast/v1', completed=True, exit_status=0,
        case=dict(files={'init.nc':str(initial)}), init_derivative_authorities=[], reference_labels=[],
        inputs={'init.nc':module.sha(initial)}, resolved_configuration=dict(files={
            n:module.sha(source/n) for n in ('namelist.atmosphere','streams.atmosphere','field-list')}))
    if defect == 'incomplete': native['completed'] = False
    if defect == 'banner': (source/'log.atmosphere.0000.out').write_text('native STOP before completion')
    (source/'MANIFEST.json').write_text(json.dumps(native))
    (source/'PUBLICATION.json').write_text(json.dumps(dict(files={
        p.name:module.sha(p) for p in source.iterdir() if p.is_file()})))
    request = dict(forecast=dict(path=str(source),publication_sha256=module.sha(source/'PUBLICATION.json')),
        inputs=inputs,auxiliary_inputs={'field-list':ref(source/'field-list')})
    if defect == 'publication-bytes': (source/'out.nc').write_bytes(b'changed output')
    if defect == 'init-bytes': initial.write_bytes(b'changed input')
    if defect == 'field-list': request['auxiliary_inputs'] = {}
    with pytest.raises(ValueError): module.generate(request, tmp_path/'must-not-execute-comparator')


@pytest.mark.parametrize('defect', ['object-flags', 'precision', 'diagnostic', 'uncaptured-command',
                                  'stale-command', 'archive-copy', 'group', 'source-pin', 'utility',
                                  'table-recipe', 'missing-table', 'missing-build', 'changed-bytes'])
def test_complete_native_authority_verifier_rejects_untrusted_recipe_or_member(tmp_path, monkeypatch, defect):
    import hashlib
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_native_authority', ROOT/'verify_native_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def digest_json(data):
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    def write(path, data):
        path.write_text(json.dumps(data)+'\n')
    def seal(directory):
        write(directory/'PUBLICATION.json', dict(files={str(p.relative_to(directory)):module.sha(p)
              for p in directory.rglob('*') if p.is_file() and p.name!='PUBLICATION.json'}, source='transport copy'))
        return module.sha(directory/'PUBLICATION.json')
    primary = dict(fork='primary fork', workflow='workflow pin', submodules={'path':'module pin'},
                   mpas_data='data pin', precision='single', musica=False)
    pins = dict(primary=primary, secondary_fork='secondary fork', pin_set_hash=digest_json(primary),
                native_builds={}, native_tables={})
    fixtures = {}
    choices = [('primary',recipe,core) for recipe in ('gnu-strict','gnu-o3')
               for core in ('atmosphere','init_atmosphere')] + [('secondary','gnu-strict','atmosphere')]
    for reference,recipe,core in choices:
        key='/'.join((reference,recipe,core));directory=tmp_path/key;directory.mkdir(parents=True)
        source=dict(primary, fork=primary['fork'] if reference=='primary' else pins['secondary_fork'])
        groups=['physics_mmm','physics_wrf','physics_noahmp']
        groups+=['physics_noaa/'+n for n in ('TEMPO','MYNN-EDMF','MYNN-SFC','RUCLSM','GFL','SMOKE','UGWP')]
        names=['src/'+g+'/unit_'+str(i)+'.o' for i,g in enumerate(groups)] if core=='atmosphere' else ['src/framework/unit.o']
        objects=[];records=[]
        for name in names:
            args=['-O3','-DSINGLE_PRECISION','-c',Path(name).with_suffix('.F').name]
            if recipe=='gnu-strict':args+=list(module.STRICT)
            command=dict(cwd=str(Path('/historical/compiler/tree')/Path(name).parent),compiler='mpif90',argv=args,status=0)
            records.append(command)
            objects.append(dict(object=name,flags_ok=True,sha256=hashlib.sha256(name.encode()).hexdigest(),
                                commands=[command],archive_copy_origins=[]))
        audit=dict(all_ok=True,errors=[],recipe=recipe,core=core,objects=objects,
                   groups={g:1 for g in groups} if core=='atmosphere' else {})
        (directory/(core+'_model')).write_bytes(key.encode())
        (directory/'toolchain.txt').write_text('original toolchain '+key)
        (directory/'build-source.diff').write_text('build flags only')
        utilities={}
        if core=='atmosphere':
            for utility in ('build_tables','build_tables_tempo_v3'):
                (directory/utility).write_text(key+utility);utilities[utility]=module.sha(directory/utility)
        manifest=dict(schema='mpas.rrfs-native-build/v1',reference=reference,recipe=recipe,core=core,
                      pin_set=source,pin_set_hash=digest_json(source),objects=len(objects),groups=audit['groups'],
                      binary_sha256=module.sha(directory/(core+'_model')),
                      toolchain_sha256=module.sha(directory/'toolchain.txt'),flags_sha256='',
                      source_diff_sha256=module.sha(directory/'build-source.diff'),utility_sha256=utilities)
        fixtures[key]=(directory,manifest,source,audit,records)
    def render_build(key):
        directory,manifest,source,audit,records=fixtures[key]
        write(directory/'SOURCE-PINS.json',source);write(directory/'flag-audit.json',audit)
        (directory/'compile-commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        manifest['flags_sha256']=module.sha(directory/'flag-audit.json');write(directory/'MANIFEST.json',manifest)
        pins['native_builds'][key]=dict(path=str(directory),publication_sha256=seal(directory),
              **{k:manifest[k] for k in ('binary_sha256','toolchain_sha256','flags_sha256','utility_sha256')})
    for key in fixtures:render_build(key)
    table_manifests={}
    for kind,utility in [('tempo','build_tables_tempo_v3'),('thompson','build_tables')]:
        for recipe in ('gnu-strict','gnu-o3'):
            name=kind+'-'+recipe;directory=tmp_path/name;directory.mkdir()
            build=pins['native_builds']['primary/'+recipe+'/atmosphere']
            (directory/'native-table').write_text(name)
            manifest=dict(schema='mpas.rrfs-'+kind+'-native-tables/v1',recipe=recipe,completed=True,exit_status=0,
              ranks=1,host='original native host',pin_set_hash=pins['pin_set_hash'],
              binary_sha256=build['utility_sha256'][utility],audit_sha256=build['flags_sha256'],
              toolchain_sha256=build['toolchain_sha256'],outputs={'native-table':module.sha(directory/'native-table')})
            table_manifests[name]=(directory,manifest)
    def render_table(name):
        directory,manifest=table_manifests[name];write(directory/'MANIFEST.json',manifest)
        pins['native_tables'][name]=dict(path=str(directory),publication_sha256=seal(directory),table_sha256=manifest['outputs'])
    for name in table_manifests:render_table(name)
    document=tmp_path/'PINS.json';write(document,dict(rrfs=pins))
    result=module.verify(document)
    assert result['verified'] and result['objects']==32 and result['table_files']==4
    assert not result['weather_execution'] and not result['physics_replay'] and not result['lane_acceptance']
    key='primary/gnu-strict/atmosphere';directory,manifest,source,audit,records=fixtures[key]
    if defect in ('object-flags','precision','diagnostic','uncaptured-command','stale-command','archive-copy','group'):
        command=audit['objects'][0]['commands'][0]
        if defect=='object-flags':
            command['argv'].append('-ffp-contract=fast');message='compiler recipe flags differ'
        elif defect=='precision':
            command['argv'].remove('-DSINGLE_PRECISION');message='single precision missing'
        elif defect=='diagnostic':
            command['argv'].append('-finit-integer=-7');message='diagnostic flags in pristine'
        elif defect=='uncaptured-command':
            records.pop(0);message='not final captured invocation'
        elif defect=='stale-command':
            records.append(dict(command,argv=command['argv']+['-ffast-math']));message='not final captured invocation'
        elif defect=='archive-copy':
            origin=audit['objects'][0]
            audit['objects'].append(dict(origin,object='src/core_atmosphere/libphys/'+Path(origin['object']).name,
                sha256='changed object bytes',archive_copy_origins=[origin['object']]))
            manifest['objects']+=1;message='archive object copy differs'
        else:
            audit['groups']['physics_noaa/TEMPO']=2;message='physics group coverage differs'
        render_build(key)
    elif defect=='source-pin':
        source['fork']='other source';message='native build pin digest differs';render_build(key)
    elif defect=='utility':
        pins['native_builds'][key]['utility_sha256']=dict(manifest['utility_sha256'],build_tables='changed utility')
        message='source diff or table utility binding differs'
    elif defect=='table-recipe':
        name='tempo-gnu-strict';table_manifests[name][1]['recipe']='gnu-o3';render_table(name)
        message='native table recipe/build authority differs'
    elif defect=='missing-table':
        pins['native_tables'].pop('tempo-gnu-strict');message='native table recipe matrix incomplete'
    elif defect=='missing-build':
        pins['native_builds'].pop('secondary/gnu-strict/atmosphere');message='native build matrix incomplete'
    else:
        (directory/'atmosphere_model').write_bytes(b'changed after publication');message='published member changed'
    write(document,dict(rrfs=pins))
    with pytest.raises(ValueError,match=message):module.verify(document)


@pytest.mark.parametrize('defect', ['changed-bytes', 'wrong-soil', 'wrong-static', 'nonfinite',
                                  'wrong-source', 'stop-zero', 'missing-output-member', 'wrong-date',
                                  'wrong-static-soil', 'wrong-stage-host'])
def test_native_case_companion_verifier_checks_real_authority_chain(tmp_path, monkeypatch, defect):
    import hashlib
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_case_verifier', ROOT/'verify_case_companions.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def write(path, data):
        path.write_text(json.dumps(data)+'\n')
    def seal(directory, omit=()):
        members = {str(p.relative_to(directory)):module.sha(p) for p in directory.rglob('*')
                   if p.is_file() and p.name!='PUBLICATION.json' and p.name not in omit}
        write(directory/'PUBLICATION.json', dict(files=members))
        return module.sha(directory/'PUBLICATION.json')
    build = tmp_path/'build'; build.mkdir()
    pins = dict(fork='original fork', workflow='original workflow', submodules={'module':'pin'},
                mpas_data='data pin', precision='single', musica=False)
    pin_hash = hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
    (build/'init_atmosphere_model').write_bytes(b'original strict initializer')
    audit = dict(all_ok=True, errors=[], core='init_atmosphere', recipe='gnu-strict', objects=[{}])
    write(build/'flag-audit.json', audit); write(build/'SOURCE-PINS.json', pins)
    native = dict(core='init_atmosphere', reference='primary', recipe='gnu-strict', pin_set=pins,
                  pin_set_hash=pin_hash, objects=1, binary_sha256=module.sha(build/'init_atmosphere_model'),
                  flags_sha256=module.sha(build/'flag-audit.json'))
    write(build/'MANIFEST.json', native); build_sha=seal(build)
    catalog=tmp_path/'catalog'; catalog.mkdir()
    originals={}; stages=[]
    for stage, output in [('static','static.nc'), ('invariant','invariant.nc'), ('case7','init.nc')]:
        directory=tmp_path/stage; directory.mkdir()
        (directory/output).write_bytes(stage.encode())
        manifest=dict(completed=True, exit_status=0, ranks=1, host='native host',
                      binary_sha256=native['binary_sha256'], audit_sha256=native['flags_sha256'],
                      source_pins=dict(pins), outputs={output:module.sha(directory/output)},
                      inputs={}, row=dict(variables=dict(init_case=7, nsoillevels=4, nvertlevels=59,
                                                        start_time='request clock')))
        write(directory/'MANIFEST.json', manifest)
        originals[stage]=(directory,manifest)
        stages.append(dict(stage=stage,path=str(directory),publication_sha256='',outputs=manifest['outputs']))
    originals['case7'][1]['inputs']['static.nc']=originals['static'][1]['outputs']['static.nc']
    finite=dict(schema='mpas.rrfs-native-finiteness-summary/v1', outputs=originals['case7'][1]['outputs'],
                files=1, fields=[dict(file='init.nc',field='real native field',words=3,nonfinite=0,
                                     first_nonfinite=None)], floating_words=3, nonfinite_words=0, all_finite=True)
    product=dict(mesh='request mesh',soil_levels=4,model_levels=59,interface_levels=60,
                 finite_floating_words=3,stages=stages)
    index=dict(date='request clock', initializer_build=str(build),initializer_publication_sha256=build_sha,pin_set_hash=pin_hash,
               products=[product],completed_stages=3,nc_files=3,finite_case7_words=3,scope='pristine inputs only')
    def render(omit=()):
        for row in stages:
            directory,manifest=originals[row['stage']]
            write(directory/'MANIFEST.json',manifest)
            row['publication_sha256']=seal(directory,omit if row['stage']=='case7' else ())
            copied=catalog/row['stage'];copied.mkdir(exist_ok=True)
            shutil.copy2(directory/'MANIFEST.json',copied/'MANIFEST.json')
            shutil.copy2(directory/'PUBLICATION.json',copied/'ORIGINAL-PUBLICATION.json')
        write(catalog/'FINITENESS.json',finite);write(catalog/'INDEX.json',index)
        return seal(catalog)
    expected=render()
    result=module.verify(catalog/'PUBLICATION.json',expected)
    assert result['verified'] and result['finite_case7_words']==3 and not result['physics_replay']
    if defect=='changed-bytes':
        (originals['case7'][0]/'init.nc').write_bytes(b'changed output')
        message='published member changed'
    else:
        if defect=='wrong-soil':
            product['soil_levels']=9;message='soil/vertical companion differs'
        elif defect=='wrong-static':
            originals['case7'][1]['inputs']['static.nc']='unrelated static';message='static-to-case7 dependency differs'
        elif defect=='nonfinite':
            finite['fields'][0]['nonfinite']=1;message='nonfinite native case7 words'
        elif defect=='wrong-source':
            originals['case7'][1]['source_pins']['fork']='other fork';message='source authority differs'
        elif defect=='stop-zero':
            originals['case7'][1]['completed']=False;message='did not complete'
        elif defect=='wrong-date':
            index['date']='other clock';message='clock differs from indexed companion'
        elif defect=='wrong-static-soil':
            originals['static'][1]['row']['variables']['nsoillevels']=9;message='stage host/clock/levels differ'
        elif defect=='wrong-stage-host':
            originals['invariant'][1]['host']='other host';message='stage host/clock/levels differ'
        else:
            message='not a published member'
        expected=render(('init.nc',) if defect=='missing-output-member' else ())
    with pytest.raises(ValueError,match=message):
        module.verify(catalog/'PUBLICATION.json',expected)


@pytest.mark.parametrize('defect', ['changed-ingress', 'missing-membership', 'primary-substitution', 'changed-runtime'])
def test_checked_ingress_contract_refuses_unwitnessed_substitution(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_diagnostic_ingress', ROOT/'verify_diagnostic_contract.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def write(path, data):
        path.write_text(json.dumps(data)+'\n')
    def seal(directory):
        data = dict(files={p.name:module.sha(p) for p in directory.iterdir()
                          if p.is_file() and p.name!='PUBLICATION.json'})
        write(directory/'PUBLICATION.json', data)
        return module.sha(directory/'PUBLICATION.json')
    shared = tmp_path/'original'; shared.mkdir()
    (shared/'base').write_bytes(b'original base deck')
    (shared/'field.nc').write_bytes(b'original stored native bytes')
    shared_sha = seal(shared)
    def reference(name):
        p = shared/name
        return dict(path=str(p), sha256=module.sha(p), bytes=p.stat().st_size,
                    publication=str(shared/'PUBLICATION.json'), publication_sha256=shared_sha)
    checked = tmp_path/'checked'; checked.mkdir()
    (checked/'atmosphere_model').write_bytes(b'diagnostic executable')
    diagnostic = dict(reference_binary=False, native_source_unchanged=True, precision='single',
        binary_sha256=module.sha(checked/'atmosphere_model'),
        checked_fortran_flags=['-fcheck=all','-finit-real=snan'])
    write(checked/'DIAGNOSTIC-MANIFEST.json',diagnostic)
    checked_sha = seal(checked)
    contract = tmp_path/'contract'; contract.mkdir()
    (contract/'deck').write_bytes(b'original generated deck')
    (contract/'partition').write_bytes(b'0\n1\n')
    row = dict(id='measured', original_base_namelist=reference('base'),
        ingress={'field.nc':reference('field.nc')}, generated_namelist='deck',
        namelist_sha256=module.sha(contract/'deck'),
        mpi2_partition=dict(path='partition',sha256=module.sha(contract/'partition')))
    index = dict(schema='mpas.rrfs-original-measured-gfl-checked-ingress-contract/v1',
        contracts=[row], checked_build=str(checked), checked_publication_sha256=checked_sha,
        checked_binary_sha256=diagnostic['binary_sha256'], recorded_runner_matches_commit=False)
    write(contract/'INDEX.json',index); contract_sha = seal(contract)
    runtime = tmp_path/'runtime'; runtime.mkdir()
    shutil.copy2(shared/'field.nc',runtime/'field.nc')
    shutil.copy2(contract/'deck',runtime/'namelist.atmosphere')
    assert module.verify(contract/'PUBLICATION.json',contract_sha,runtime,'measured',1)['verified']
    if defect=='changed-ingress':
        (shared/'field.nc').write_bytes(b'changed native stored bytes')
        message = 'shared ingress bytes changed'
    elif defect=='missing-membership':
        data = json.loads((shared/'PUBLICATION.json').read_text()); del data['files']['field.nc']
        write(shared/'PUBLICATION.json',data)
        for item in [row['original_base_namelist'],*row['ingress'].values()]:
            item['publication_sha256']=module.sha(shared/'PUBLICATION.json')
        message = 'not a published member'
    elif defect=='primary-substitution':
        diagnostic['reference_binary']=True
        write(checked/'DIAGNOSTIC-MANIFEST.json',diagnostic)
        index['checked_publication_sha256']=seal(checked)
        message = 'labelled source-unchanged checked diagnostic'
    else:
        (runtime/'namelist.atmosphere').write_bytes(b'altered scheme deck')
        message = 'runtime ingress differs'
    write(contract/'INDEX.json',index); contract_sha=seal(contract)
    with pytest.raises(ValueError,match=message):
        module.verify(contract/'PUBLICATION.json',contract_sha,runtime,'measured',1)


def test_archived_forecast_readonly_ingress_rejects_native_output_collisions():
    spec = importlib.util.spec_from_file_location('rrfs_archive_roles', ROOT / 'archive_outputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    streams = ET.fromstring('''<streams>
      <stream type="input" filename_template="lbc.$Y-$M-$D_$h.$m.$s.nc"/>
      <stream type="input;output" filename_template="restart.$Y-$M-$D_$h.$m.$s.nc"/>
      <stream type="input" filename_template="init.nc"/>
      <stream type="output" filename_template="init.nc"/>
    </streams>''')
    assert module.readonly_inputs(streams, ['lbc.2023-03-31_12.00.00.nc']) == {'lbc.2023-03-31_12.00.00.nc'}
    for name in ['init.nc', 'restart.2023-03-31_12.00.00.nc', 'unknown.nc', 'lbc.not-a-native-clock.nc']:
        with pytest.raises(ValueError, match='exclusively a native input'):
            module.readonly_inputs(streams, [name])


def test_archived_forecast_template_matching_does_not_admit_extra_records():
    spec = importlib.util.spec_from_file_location('rrfs_archive_templates', ROOT / 'archive_outputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    template = 'output.$Y-$M-$D_$h.$m.$s.nc'
    assert module.matches(template, 'output.2023-03-31_12.00.00.nc')
    for name in ['output.2023-03-31_12X00X00.nc', 'output.23-03-31_12.00.00.nc', 'output.2023-03-31_12.00.00.nc.extra', '../output.2023-03-31_12.00.00.nc']:
        assert not module.matches(template, name)


def test_archive_clock_plan_retains_native_initial_da_and_checkpoint_companions():
    spec = importlib.util.spec_from_file_location('rrfs_archive_da_clocks', ROOT / 'archive_outputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    streams=ET.fromstring('''<streams><stream type="input;output" input_interval="initial_only"
      filename_template="mpasout.$Y-$M-$D_$h.$m.$s.nc" output_timelevels="0 12 24"/></streams>''')
    expected=module.expected_outputs({'start':'2023-03-31_12:00:00','duration':'1_00:00:00'},streams,[])
    assert expected=={'mpasout.2023-03-31_12.00.00.nc':['2023-03-31_12:00:00'],
                     'mpasout.2023-04-01_00.00.00.nc':['2023-04-01_00:00:00'],
                     'mpasout.2023-04-01_12.00.00.nc':['2023-04-01_12:00:00']}


def test_recipe_measurement_preserves_explicit_generated_table_difference():
    spec = importlib.util.spec_from_file_location('rrfs_recipe_contract', ROOT / 'compare_recipes.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    common = dict(completed=True, exit_status=0, source_inputs_unchanged=True,
                  schema='native', pin_set_hash='pins', fork='fork', deck='deck',
                  deck_index_sha256='index', resolved_configuration={'physics': 'original'},
                  runtime={'duration': '24h'}, selected_table_overrides={}, ranks=1, host='same-host')
    recipes = [dict(recipe='strict', binary_sha256='strict-binary', toolchain_sha256='strict-flags'),
               dict(recipe='o3', binary_sha256='o3-binary', toolchain_sha256='o3-flags')]
    manifests = [dict(common, **recipe, inputs={'init.nc': 'same-weather', 'table': recipe['recipe']})
                 for recipe in recipes]
    authority = dict(recipes=recipes, recipe_input_differences={'table': ['strict', 'o3']})
    assert module.validate(manifests, authority) == authority['recipe_input_differences']
    manifests[1]['inputs']['init.nc'] = 'changed-weather'
    with pytest.raises(ValueError, match='unwitnessed recipe input'):
        module.validate(manifests, authority)
    manifests[1]['inputs']['init.nc'] = 'same-weather'
    manifests[1]['runtime'] = {'duration': '12h'}
    with pytest.raises(ValueError, match='scientific forecast contract differs'):
        module.validate(manifests, authority)


def test_recipe_measurement_rejects_changed_binary_or_missing_native_input():
    spec = importlib.util.spec_from_file_location('rrfs_recipe_authority', ROOT / 'compare_recipes.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    common = dict(completed=True, exit_status=0, source_inputs_unchanged=True,
                  schema='native', pin_set_hash='pins', fork='fork', deck='deck', deck_index_sha256='index',
                  resolved_configuration={}, runtime={}, selected_table_overrides={}, ranks=1, host='same-host')
    recipes = [dict(recipe=r, binary_sha256=r, toolchain_sha256=r) for r in ('strict', 'o3')]
    manifests = [dict(common, **recipe, inputs={'table': recipe['recipe']}) for recipe in recipes]
    authority = dict(recipes=recipes, recipe_input_differences={'table': ['strict', 'o3']})
    manifests[1]['binary_sha256'] = 'changed'
    with pytest.raises(ValueError, match='recipe build authority changed'):
        module.validate(manifests, authority)
    manifests[1]['binary_sha256'] = 'o3'
    manifests[1]['inputs'] = {}
    authority['recipe_input_differences'] = {'table': ['strict', None]}
    with pytest.raises(ValueError, match='cannot omit an input'):
        module.validate(manifests, authority)


def test_published_deck_inputs_reject_changed_and_unsealed_original_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_published_inputs', ROOT / 'run_published_decks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sha = module.replay.load('run_forecast').sha
    original = tmp_path / 'init.nc'; original.write_bytes(b'original native initializer bytes')
    pub = tmp_path / 'PUBLICATION.json'
    pub.write_text(json.dumps(dict(files={original.name:sha(original)})))
    pin = sha(pub)
    assert module.authenticate(tmp_path, pin) == {original.name:sha(original)}
    original.write_bytes(b'changed input')
    with pytest.raises(ValueError, match='member bytes changed'):
        module.authenticate(tmp_path, pin)
    with pytest.raises(ValueError, match='unsafe or unsealed'):
        module.authenticate(tmp_path, pin, ['../init.nc'])
    with pytest.raises(ValueError, match='unsafe or unsealed'):
        module.authenticate(tmp_path, pin, ['not-sealed.nc'])
    with pytest.raises(ValueError, match='publication authority changed'):
        module.authenticate(tmp_path, '0'*64)


def test_published_deck_launch_rejects_changed_request_before_native_attempt(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_published_launch', ROOT / 'run_published_decks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    authority = tmp_path / 'REQUEST.json'; authority.write_bytes(b'changed')
    out = tmp_path / 'native-attempt'
    with pytest.raises(ValueError, match='request authority changed'):
        module.run(dict(work_root=str(tmp_path), input_hashes={str(authority):'0'*64}), out)
    assert not out.exists()


def test_published_replays_bind_checkpoint_and_comparison_to_same_recipe():
    spec = importlib.util.spec_from_file_location('rrfs_bound_replays', ROOT / 'run_published_decks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = [dict(name='deck-' + recipe, inputs={'init.nc': '/original'}, ranks=1,
                 overlay={'settings': {'physics_choice': 'original'},
                          'streams': {'da_state': {'filename_template': 'original'}}})
            for recipe in ('strict', 'o3')]
    phases = [dict(name='cold', ranks=1),
              dict(name='resume', ranks=1,
                   overlay={'settings': {'config_do_restart': True},
                            'streams': {'da_state': {'output_timelevels': '12 24'}}},
                   from_outputs={'cold': {'restart.nc': 'restart.nc'}}),
              dict(name='repeat', ranks=1, compare={'resume': 'rerun'})]
    result = module.expand_replays(rows, phases)
    for offset, recipe in ((0, 'strict'), (3, 'o3')):
        resumed, repeated = result[offset + 1:offset + 3]
        assert resumed['from_outputs'] == {'deck-' + recipe + '-cold': {'restart.nc': 'restart.nc'}}
        assert repeated['compare'] == {'deck-' + recipe + '-resume': 'rerun'}
        assert resumed['overlay']['settings']['physics_choice'] == 'original'
        assert resumed['overlay']['streams']['da_state']['filename_template'] == 'original'
    assert all('config_do_restart' not in row['overlay']['settings'] for row in rows)


def test_published_replays_refuse_unwitnessed_dependencies():
    spec = importlib.util.spec_from_file_location('rrfs_replay_dependencies', ROOT / 'run_published_decks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='earlier same-deck phase'):
        module.expand_replays([], [dict(name='resume', ranks=1, from_outputs={'other-deck': {}})])
    with pytest.raises(ValueError, match='unique simple'):
        module.expand_replays([], [dict(name='../outside', ranks=24)])
    with pytest.raises(ValueError, match='unique simple'):
        module.expand_replays([], [dict(name='cold', ranks=True)])


def test_published_rank_replay_refuses_changed_partition_or_weather_replacement(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_rank_partition', ROOT / 'run_published_decks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    partition = tmp_path / 'partition'; partition.write_bytes(b'0\n1\n')
    authority = dict(path=str(partition), sha256=module.replay.load('run_forecast').sha(partition))
    row = dict(name='deck-strict', inputs={'init.nc': '/original'}, overlay={}, ranks=1)
    phase = dict(name='ranked', ranks=24, input_files={'graph.part.24': authority})
    ranked = module.expand_replays([row], [phase])[0]
    assert ranked['ranks'] == 24 and ranked['inputs']['graph.part.24'] == str(partition)
    with pytest.raises(ValueError, match='replace initialized weather'):
        module.expand_replays([row], [dict(phase, input_files={'init.nc': authority})])
    partition.write_bytes(b'changed partition')
    with pytest.raises(ValueError, match='partition input authority changed'):
        module.expand_replays([row], [phase])


def test_sparse_publication_preserves_native_bytes_and_seals_original_inventory(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_sparse_publication', ROOT / 'publish_sparse.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'source'; source.mkdir()
    native = source / 'original.nc'
    native.write_bytes(b'nonzero prefix' + b'\0' * (4 * 1024**2) + b'nonzero suffix')
    ready = tmp_path / 'READY.md'; ready.write_text('')
    destination = tmp_path / 'immutable'
    publication = module.publish(source, destination, ['original.nc'], ready, 'native byte preservation')
    assert publication['files'] == 2
    assert (destination / native.name).read_bytes() == native.read_bytes()
    assert (destination / native.name).stat().st_blocks < native.stat().st_blocks
    assert not destination.stat().st_mode & 0o222
    manifest = json.loads((destination / 'PUBLICATION.json').read_text())
    assert manifest['files'][native.name] == module.base.sha(native)
    recipe = json.loads((destination / 'SPARSE-COPY-RECIPE.json').read_text())
    assert recipe['every_copied_file_sha256_checked'] and not recipe['weather_arrays_transformed']
    assert publication['publication_sha256'] in ready.read_text()


def test_case_endpoint_retirement_refuses_changed_or_unlisted_native_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_pipeline', ROOT / 'run_case_pipeline.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cache = module.load('cache_cases')
    endpoint = tmp_path / 'private-endpoint'; endpoint.mkdir()
    native = endpoint / 'GFS:time'; native.write_bytes(b'original native intermediate')
    publication = endpoint / 'PUBLICATION.json'
    publication.write_text(json.dumps({'files':{native.name:cache.sha(native)}}))
    digest = cache.sha(publication)
    native.write_bytes(b'changed native bytes')
    with pytest.raises(ValueError, match='endpoint bytes changed'):
        module.retire_endpoint(endpoint, digest, tmp_path)
    assert endpoint.exists()
    native.write_bytes(b'original native intermediate')
    extra = endpoint / 'unlisted'; extra.write_bytes(b'not owned by receipt')
    with pytest.raises(ValueError, match='inventory differs'):
        module.retire_endpoint(endpoint, digest, tmp_path)
    assert native.exists() and extra.exists()
    extra.unlink(); endpoint.chmod(0o555)
    module.retire_endpoint(endpoint, digest, tmp_path)
    assert not endpoint.exists()


def test_received_case_retirement_authenticates_all_outputs_before_deletion(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_receiver', ROOT / 'receive_case.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    scratch = tmp_path / 'scratch'; source = scratch / 'transfer'; source.mkdir(parents=True)
    archive = tmp_path / 'archive'; archive.mkdir()
    outputs = {}
    for name in ('init.nc', 'lbc.nc'):
        (source / name).write_bytes(b'actual received native bytes')
        shutil.copy2(source / name, archive / name)
        outputs[name] = module.cache.sha(source / name)
    (source / 'MANIFEST.json').write_text(json.dumps({'outputs':outputs}))
    (archive / 'PUBLICATION.json').write_text(json.dumps({'files':outputs}))
    publication = {'publication_sha256':module.cache.sha(archive / 'PUBLICATION.json')}
    (archive / 'lbc.nc').write_bytes(b'changed archive')
    with pytest.raises(ValueError, match='immutable archive'):
        module.retire_transferred_outputs(source, archive, scratch, publication)
    assert all((source / name).exists() for name in outputs)
    shutil.copy2(source / 'lbc.nc', archive / 'lbc.nc')
    with pytest.raises(ValueError, match='owned transfer scratch'):
        module.retire_transferred_outputs(source, archive, tmp_path / 'archive', publication)
    receipt = module.retire_transferred_outputs(source, archive, scratch, publication)
    assert {row['file'] for row in receipt['retired']} == set(outputs)
    assert not any((source / name).exists() for name in outputs)
    assert (source / 'MANIFEST.json').exists()
    assert all((archive / name).exists() for name in outputs)


def test_finiteness_uses_required_actual_rust_counts_and_complete_inventory():
    import copy
    spec = importlib.util.spec_from_file_location('rrfs_finiteness', ROOT / 'summarize_finiteness.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    field = dict(field='temperature', netcdf_type=5, words=2, nan=0,
                 positive_infinity=0, negative_infinity=0, first_nonfinite=None)
    inventory = dict(schema='mpas.rrfs-native-field-finiteness/v1', fields=[field])
    outputs = {'output.nc':'original-output-sha'}
    assert module.summarize(outputs, {'output.nc':inventory})['all_finite']
    for key in ('nan', 'positive_infinity', 'negative_infinity'):
        changed = copy.deepcopy(inventory); changed['fields'][0][key] = 1
        changed['fields'][0]['first_nonfinite'] = [1]
        result = module.summarize(outputs, {'output.nc':changed})
        assert not result['all_finite'] and result['nonfinite_words'] == 1
        del changed['fields'][0][key]
        with pytest.raises(KeyError): module.summarize(outputs, {'output.nc':changed})
    with pytest.raises(ValueError, match='every emitted'):
        module.summarize({'output.nc':'sha', 'da.nc':'sha'}, {'output.nc':inventory})
    changed = copy.deepcopy(inventory); changed['fields'][0]['nan'] = 1
    with pytest.raises(ValueError, match='inconsistent'):
        module.summarize(outputs, {'output.nc':changed})


def test_geography_image_publication_authenticates_original_and_transferred_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_geography_receiver', ROOT / 'receive_geography_images.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'transfer'; source.mkdir()
    original = tmp_path / 'original.tar.gz'; original.write_bytes(b'original pinned archive')
    image = source / 'native.sqfs'; image.write_bytes(b'native verified image')
    manifest = source / 'MANIFEST.json'
    receipt = dict(completed=True, storage='native-read-only-compressed-filesystem',
                   archives=[dict(sha256=module.sha(original), bytes=original.stat().st_size)],
                   products=[dict(source_sha256=module.sha(original), image=image.name,
                                  image_sha256=module.sha(image), bytes=image.stat().st_size,
                                  all_regular_members_native_cmp_identical=True)])
    manifest.write_text(json.dumps(receipt))
    expected = dict(manifest_sha256=module.sha(manifest),
                    archive_paths={module.sha(original):str(original)}, description='verified native images')
    destination = tmp_path / 'published'; ready = tmp_path / 'READY.md'
    image.write_bytes(b'changed transferred image')
    with pytest.raises(ValueError, match='verified image changed'):
        module.receive(source, destination, ready, expected)
    image.write_bytes(b'native verified image'); original.write_bytes(b'changed original source')
    with pytest.raises(ValueError, match='verified image changed'):
        module.receive(source, destination, ready, expected)
    original.write_bytes(b'original pinned archive')
    result = module.receive(source, destination, ready, expected)
    assert result['files'] == 2 and (destination / image.name).read_bytes() == b'native verified image'
    assert not destination.stat().st_mode & 0o222
    assert result['publication_sha256'] in ready.read_text()


def fixture(root, argv, compiler='mpif90', status=0, path='src/core_init_atmosphere/test.o'):
    obj = root / path
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.touch()
    row = dict(cwd=str(obj.parent), compiler=compiler, argv=argv, status=status)
    with (root / 'compile-commands.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def good(source='test.F90'):
    return ['-O3', *flags.STRICT, '-DSINGLE_PRECISION', '-c', source]


def test_audit_matches_directory_and_source_output(tmp_path):
    fixture(tmp_path, good())
    fixture(tmp_path, ['-O3', '-DSINGLE_PRECISION', '-c', 'test.F90'], path='src/framework/test.o')
    result = flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')
    assert result['errors'] == ['src/framework/test.o']


def test_missing_capture_failed_compile_precision_and_unsafe_flags(tmp_path):
    fixture(tmp_path, good())
    assert flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    for args, status in [(good(), 1), ([a for a in good() if a != '-DSINGLE_PRECISION'], 0),
                         (good() + ['-ffp-contract=fast'], 0)]:
        (tmp_path / 'compile-commands.jsonl').unlink()
        fixture(tmp_path, args, status=status)
        assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    (tmp_path / 'src/framework').mkdir()
    (tmp_path / 'src/framework/uncaptured.o').touch()
    assert 'src/framework/uncaptured.o' in flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['errors']


def test_every_noaa_group_required_for_atmosphere(tmp_path):
    fixture(tmp_path, good())
    result = flags.audit(tmp_path, 'gnu-strict', 'atmosphere')
    for name in ('TEMPO', 'MYNN-EDMF', 'MYNN-SFC', 'RUCLSM', 'GFL', 'SMOKE', 'UGWP'):
        assert 'missing group physics_noaa/' + name in result['errors']


def test_explicit_output_and_c_flags(tmp_path):
    fixture(tmp_path, ['-O3', *flags.STRICT, '-c', 'other.c', '-o', 'test.o'], compiler='mpicc')
    assert flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']


def test_incremental_rebuild_audits_last_compilation(tmp_path):
    fixture(tmp_path, ['-O3', '-DSINGLE_PRECISION', '-c', 'test.F90'])
    assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    fixture(tmp_path, good())
    assert flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    fixture(tmp_path, good(), status=1)
    assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']


@pytest.mark.parametrize('diagnostic', ['bounds-snan', 'bounds-snan-integer-poison'])
def test_full_diagnostic_flags_are_required_for_fortran_and_archive_copies(tmp_path, diagnostic):
    (tmp_path / 'SOURCE-PINS.json').write_text(json.dumps(dict(diagnostic=diagnostic)))
    origin = 'src/core_atmosphere/physics/physics_noaa/GFL/test.o'
    checked = good() + ['-g', '-fcheck=all', '-finit-real=snan']
    if diagnostic == 'bounds-snan-integer-poison':
        checked += ['-finit-integer=-2147483647']
    fixture(tmp_path, good(), path=origin)
    assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    fixture(tmp_path, checked, path=origin)
    copy = tmp_path / 'src/core_atmosphere/libphys/test.o'
    copy.parent.mkdir(); copy.write_bytes((tmp_path / origin).read_bytes())
    fixture(tmp_path, ['-O3', *flags.STRICT, '-c', 'control.c', '-o', 'control.o'],
            compiler='mpicc', path='src/framework/control.o')
    assert flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
    overrides = ['-fcheck=none', '-finit-real=zero']
    if diagnostic == 'bounds-snan-integer-poison':
        fixture(tmp_path, checked[:-1], path=origin)
        assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']
        overrides.append('-finit-integer=0')
    for override in overrides:
        fixture(tmp_path, checked + [override], path=origin)
        result = flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')
        assert result['diagnostic'] == diagnostic
        assert sorted(result['errors']) == sorted([origin, 'src/core_atmosphere/libphys/test.o'])


@pytest.mark.parametrize('diagnostic', ['bounds-snan', 'bounds-snan-integer-poison'])
def test_diagnostic_insertion_preserves_c_and_cpp_strict_flags(diagnostic):
    spec = importlib.util.spec_from_file_location('rrfs_prepare_diagnostic', ROOT / 'prepare_build.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    strict = '-O3 ' + ' '.join(flags.STRICT) + ' '
    text = '\ngfortran:\n' + ''.join('"' + kind + ' = ' + strict + '"\n'
        for kind in ('FFLAGS_OPT', 'CFLAGS_OPT', 'CXXFLAGS_OPT')) + '\ngfortran-clang:\nunchanged\n'
    changed = module.add_diagnostic_flags(text, diagnostic)
    for kind in ('CFLAGS_OPT', 'CXXFLAGS_OPT'):
        assert '"' + kind + ' = ' + strict + '"' in changed
    assert changed.count('-fcheck=all') == changed.count('-finit-real=snan') == 1
    assert changed.count('-finit-integer=-2147483647') == (diagnostic == 'bounds-snan-integer-poison')
    assert changed.endswith('\ngfortran-clang:\nunchanged\n')
    with pytest.raises(ValueError, match='diagnostic flag location'):
        module.add_diagnostic_flags(text.replace('FFLAGS_OPT', 'OTHER_FLAGS'), diagnostic)


def test_diagnostic_source_witness_checks_actual_pinned_git_bytes(tmp_path):
    import subprocess
    spec = importlib.util.spec_from_file_location('rrfs_checked_publisher', ROOT / 'publish_diagnostic_build.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, build = tmp_path / 'source', tmp_path / 'build'
    source.mkdir(); build.mkdir()
    (source / 'module.F90').write_text('module original\nend module\n')
    (build / 'module.F90').write_bytes((source / 'module.F90').read_bytes())
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', 'module.F90'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Native diagnostic test',
                    '-c', 'user.email=diagnostic@example.invalid', 'commit', '-qm', 'Pinned fixture'], check=True)
    pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    pins = dict(fork=pin, submodules={})
    assert module.verify_sources(source, build, pins)['all_pinned_code_bytes_identical']
    (build / 'module.F90').write_text('module repaired\nend module\n')
    with pytest.raises(ValueError, match='source bytes changed'):
        module.verify_sources(source, build, pins)
    (source / 'module.F90').write_text('module altered\nend module\n')
    with pytest.raises(ValueError, match='repository differs'):
        module.verify_sources(source, build, pins)


def test_pin_set_contains_all_fork_gitlinks():
    pins = json.loads((ROOT / 'pins.json').read_text())
    assert len(pins['submodules']) == 8
    assert pins['fork'] == '1d6e370e7b43a332074809f170532a216c2212fb'
    assert pins['precision'] == 'single' and not pins['musica']


@pytest.mark.parametrize('origin,archive', [('physics', 'libphys'), ('chemistry', 'libchem')])
def test_archive_extraction_requires_same_name_and_exact_object_bytes(tmp_path, origin, archive):
    fixture(tmp_path, good(), path=f'src/core_atmosphere/{origin}/test.o')
    original = tmp_path / f'src/core_atmosphere/{origin}/test.o'
    original.write_bytes(b'compiled-object')
    copy = tmp_path / f'src/core_atmosphere/{archive}/test.o'
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())
    result = flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')
    assert result['all_ok']
    row = next(r for r in result['objects'] if f'/{archive}/' in r['object'])
    assert row['archive_copy_origins'] == [f'src/core_atmosphere/{origin}/test.o']
    copy.write_bytes(b'different-object')
    assert not flags.audit(tmp_path, 'gnu-strict', 'init_atmosphere')['all_ok']


def load_decks():
    spec = importlib.util.spec_from_file_location('rrfs_decks', ROOT / 'generate_decks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_rejects_incomplete_forecasts_and_changed_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_identity', ROOT / 'identity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a, b = tmp_path / 'a', tmp_path / 'b'
    for path in (a, b):
        path.mkdir()
        (path / 'MANIFEST.json').write_text(json.dumps(dict(completed=False, exit_status=1)))
    with pytest.raises(ValueError, match='completed native forecasts'):
        module.witness(a, b, tmp_path / 'unused-comparator', 'rerun')
    keys = ('schema', 'pin_set_hash', 'fork', 'recipe', 'binary_sha256', 'toolchain_sha256',
            'deck', 'deck_index_sha256', 'resolved_configuration', 'runtime', 'selected_table_overrides')
    manifest = dict.fromkeys(keys, 'same')
    manifest.update(completed=True, exit_status=0, ranks=1, inputs={'init.nc': 'original'})
    (a / 'MANIFEST.json').write_text(json.dumps(manifest))
    manifest['inputs'] = {'init.nc': 'changed'}
    (b / 'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='weather input or table bytes differ'):
        module.witness(a, b, tmp_path / 'unused-comparator', 'rerun')


@pytest.mark.parametrize('defect', ['publication', 'body', 'unpublished-manifest',
    'unpublished-stream', 'incomplete', 'schema', 'lost-derivative-label'])
def test_published_field_recomparison_rejects_untrusted_archives(tmp_path, monkeypatch, defect):
    import hashlib
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_recompare', ROOT/'recompare_published.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path/'archive'; root.mkdir()
    (root/'output.nc').write_bytes(b'opaque archived weather bytes')
    manifest = dict(schema='mpas.rrfs-native-forecast/v1', completed=True, exit_status=0,
        outputs={'output.nc': module.sha(root/'output.nc')})
    if defect == 'unpublished-stream': manifest['outputs']['extra.nc'] = '0'*64
    elif defect == 'incomplete': manifest['completed'] = False
    elif defect == 'schema': manifest['schema'] = 'untrusted'
    elif defect == 'lost-derivative-label':
        from rrfs.ruled_reference import authority, INIT_BINARY
        manifest['init_derivative_authorities'] = [authority('init', INIT_BINARY)]
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    members = {name: module.sha(root/name) for name in ('MANIFEST.json', 'output.nc')}
    if defect == 'unpublished-manifest': members.pop('MANIFEST.json')
    (root/'PUBLICATION.json').write_text(json.dumps(dict(files=members)))
    row = dict(path=str(root), publication_sha256=module.sha(root/'PUBLICATION.json'))
    if defect == 'publication': row['publication_sha256'] = hashlib.sha256(b'different').hexdigest()
    elif defect == 'body': (root/'output.nc').write_bytes(b'changed weather bytes')
    with pytest.raises(ValueError): module.archive(row, {})


@pytest.mark.parametrize('defect', ['comparator', 'duplicate', 'unsafe-id', 'mode', 'replica', 'same-archive'])
def test_published_field_recomparison_rejects_ambiguous_proof_requests(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_recompare', ROOT/'recompare_published.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path/'archive'; root.mkdir()
    (root/'output.nc').write_bytes(b'opaque native bytes')
    (root/'MANIFEST.json').write_text(json.dumps(dict(schema='mpas.rrfs-native-forecast/v1',
        completed=True, exit_status=0, outputs={'output.nc': module.sha(root/'output.nc')})))
    (root/'PUBLICATION.json').write_text(json.dumps(dict(files={
        name: module.sha(root/name) for name in ('MANIFEST.json', 'output.nc')})))
    executable = tmp_path/'never-execute'; executable.write_bytes(b'not an executable')
    row = dict(path=str(root), publication_sha256=module.sha(root/'PUBLICATION.json'))
    comparison = dict(id='same', mode='rerun', reference=row, other=row)
    request = dict(schema='mpas.rrfs-published-field-recomparison-request/v1',
        comparator_sha256=module.sha(executable), comparisons=[comparison])
    replicas = {}
    if defect == 'comparator': request['comparator_sha256'] = '0'*64
    elif defect == 'duplicate': request['comparisons'].append(comparison.copy())
    elif defect == 'unsafe-id': comparison['id'] = '../escape'
    elif defect == 'mode': comparison['mode'] = 'unknown'
    elif defect == 'replica': replicas['undeclared'] = str(root)
    with pytest.raises(ValueError): module.recompare(request, executable, tmp_path/'result', replicas)


@pytest.mark.parametrize('defect', ['parent-member', 'absolute-member', 'symlink', 'oversized'])
def test_published_field_transport_rejects_escape_and_unbounded_members(tmp_path, monkeypatch, defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('rrfs_recompare', ROOT/'recompare_published.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path/'archive'; root.mkdir()
    outside = tmp_path/'outside'; outside.write_bytes(b'not an archive member')
    name = 'output.nc'
    if defect == 'parent-member': name = '../outside'
    elif defect == 'absolute-member': name = str(outside)
    elif defect == 'symlink': (root/name).symlink_to(outside)
    else:
        with (root/name).open('wb') as stream: stream.truncate(200*1024**2)
    (root/'PUBLICATION.json').write_text(json.dumps(dict(files={name: '0'*64})))
    row = dict(path=str(root), publication_sha256=module.sha(root/'PUBLICATION.json'))
    request = dict(schema='mpas.rrfs-published-field-recomparison-request/v1',
        comparisons=[dict(reference=row, other=row)])
    with pytest.raises(ValueError): module.stage_archives(request, tmp_path/'transport')
    assert outside.read_bytes() == b'not an archive member'


def test_relative_registry_includes_and_commented_include(tmp_path):
    module = load_decks()
    root = tmp_path / 'src/core_atmosphere'
    (root / 'diagnostics').mkdir(parents=True)
    (root / 'Registry.xml').write_text('<registry>\n#include "diagnostics/Registry.xml"\n</registry>')
    (root / 'diagnostics/Registry.xml').write_text('<!-- #include "missing.xml" -->\n#include "nested.xml"')
    (root / 'diagnostics/nested.xml').write_text('<nml_record name="physics"><nml_option name="config_flag" type="logical"/></nml_record>')
    groups, sources = module.registry(tmp_path)
    assert groups['config_flag'][0] == 'physics'
    assert len(sources) == 3
    with pytest.raises(ValueError):
        module.literal(True, 'integer')


def test_workflow_and_fork_decks_include_required_options(tmp_path):
    source = Path(os.environ.get('RRFS_REFERENCE_SOURCE') or Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs/MPAS-Model-rrfs')
    workflow = source.parent / 'rrfs-workflow'
    if not (source / 'src/core_atmosphere/Registry.xml').exists():
        pytest.skip('pinned external native sources unavailable')
    module = load_decks()
    runtime = dict(start='2023-03-31_12:00:00', duration='1_00:00:00', output_interval='01:00:00',
                   restart_interval='12:00:00', dt=720, substeps=4, mesh='x1.2562', limited_area=False,
                   graph_prefix='x1.2562.graph.info.part.', lbc_interval_hours=1, input_filenames={},
                   workflow_variables=dict(pio_num_iotasks=0, pio_stride=1, do_DAcycling='false', radt=60,
                                           lsm_scheme='sf_ruc', nsoillevels=9, jedi_da='false', do_sppt='false'))
    index = module.generate(source, workflow, tmp_path / 'decks', runtime,
                            json.loads((ROOT / 'deck-specs.json').read_text()))
    rows = {row['deck']: row for row in index['decks']}
    settings = rows['suite-hrrrv5']['resolved_settings']
    assert settings['config_lsm_scheme'] == 'sf_ruc' and settings['num_soil_layers'] == 9
    assert settings['config_tempo_ml_for_bl_nc'] is True
    assert settings['config_mynn_mixnumcon'] == 0
    assert rows['pbl_scheme-bl_myj']['companions']['config_sfclayer_scheme'] == 'sf_myj'
    assert rows['radt_cld_scheme-cld_fraction_mynn']['companions']['config_pbl_scheme'] == 'bl_mynnedmf'
    assert rows['gfl-sub3d-1']['resolved_settings']['config_gfl_sub3d'] == 1
    assert rows['hrrrv5-da-cycling']['resolved_settings']['config_do_DAcycling'] is True
    assert rows['smoke_scheme-on']['resolved_settings']['config_smoke_scheme'] == 'on'
    assert 'stream_list/stream_list.atmosphere.da_state' in rows['suite-hrrrv5']['files']
    for key in ('aerosolaware', 'hailaware', 'ml_for_bl_nc'):
        assert 'mp_tempo-tempo_' + key in rows
    for key in ('mixscalars', 'mixaerosols', 'mixnumcon'):
        assert 'bl_mynnedmf-mynn_' + key in rows
    assert rows['mp-tempo-cldfra-true']['resolved_settings']['config_radt_cld_scheme'] == 'cld_incidence'
    for option in ('ml-nc', 'ml-nr', 'melting', 'cldfra'):
        assert 'mp-tempo-' + option + '-true' in rows
        assert 'mp-tempo-' + option + '-false' in rows
    selection = json.loads((ROOT / 'deck-specs.json').read_text())
    selection['selected_names'] = ['suite-hrrrv5']
    runtime.update(do_restart=True, start='2023-04-01_00:00:00', duration='12:00:00')
    one = module.generate(source, workflow, tmp_path / 'selected-decks', runtime, selection)
    assert [row['deck'] for row in one['decks']] == ['suite-hrrrv5']
    assert one['decks'][0]['resolved_settings']['config_do_restart'] is True
    assert one['decks'][0]['resolved_settings']['config_start_time'] == '2023-04-01_00:00:00'


def test_case_matrix_requires_hourly_endpoints_and_ruc_soils(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_matrix', ROOT / 'generate_case_matrix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    levels = tmp_path / 'levels.txt'
    levels.write_text('0\n100\n200\n')
    request = dict(vertical_levels=str(levels), soil_levels=[9, 4],
                   first_guess_levels=58, first_guess_soil_levels=4, met_prefix='GFS',
                   geography=str(tmp_path / 'geog'),
                   case7_runtime_files={'QNWFA_QNIFA_SIGMA_MONTHLY.dat':'monthly-native-aerosol.dat'},
                   templates={stage:dict(namelist='namelist.'+stage, streams='streams.'+stage)
                              for stage in ('static', 'invariant', 'case7')},
                   dates=[dict(start='2023-03-31_12:00:00', forecast_hours=2,
                               hourly_gribs=[['main0','extra0'], ['main1','extra1'], ['main2','extra2']])],
                   meshes=[dict(name='regional', grid='mesh.nc', limited_area=True, ranks=1)])
    result = module.generate(request, tmp_path / 'work', tmp_path / 'ref')
    assert len(result['decode']) == 3 and len(result['cases']) == 12
    ruc = [r for r in result['cases'] if r['variables']['nsoillevels'] == 9]
    lbcs = [r for r in ruc if r['variables']['init_case'] == 9]
    assert [r['variables']['start_time'] for r in lbcs] == [
        '2023-03-31_12:00:00', '2023-03-31_13:00:00', '2023-03-31_14:00:00']
    assert all(r['variables']['lbc_hydrometeors_gfs'] == 'true' and
               r['variables']['lbc_hydrometeors_rrfs'] == 'false' for r in lbcs)
    assert all(r['stream_replacements']['input_stream'] == 'invariant.nc' and
               r['inputs']['invariant.nc'].endswith('/invariant/invariant.nc') for r in lbcs)
    assert all(r['runtime_files']['QNWFA_QNIFA_SIGMA_MONTHLY.dat'] == 'monthly-native-aerosol.dat'
               for r in lbcs)
    assert ruc[0]['variables']['nvertlevels'] == 2
    assert all(row['decoded'].endswith('/decoded/gfs-20230331-12')
               for row in result['cases'] if row['name'].endswith(('/static', '/invariant')))
    request['dates'][0]['hourly_gribs'].pop()
    with pytest.raises(ValueError, match='both forecast endpoints'):
        module.generate(request, tmp_path / 'work', tmp_path / 'ref')


def test_runtime_tables_exclude_executables_and_run_logs(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_run_tables_filter', ROOT / 'run_forecast.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('freeze_water_data_tempo_v3', 'SOILPARM.TBL', 'native.stdout.log', 'build_tables_tempo_v3'):
        (tmp_path / name).touch()
    (tmp_path / 'build_tables_tempo_v3').chmod(0o755)
    assert module.table_file(tmp_path / 'freeze_water_data_tempo_v3')
    assert module.table_file(tmp_path / 'SOILPARM.TBL')
    assert not module.table_file(tmp_path / 'native.stdout.log')
    assert not module.table_file(tmp_path / 'build_tables_tempo_v3')


def test_native_writable_input_copy_preserves_sealed_source_and_ingress(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_mutable_ingress', ROOT / 'run_forecast.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'sealed-state'; source.write_bytes(b'ingress-bytes'); source.chmod(0o444)
    table = tmp_path / 'table'; table.write_bytes(b'constant-table')
    out = tmp_path / 'run'; out.mkdir()
    files = {'state.nc':source, 'table.TBL':table}
    ingress = {name:module.sha(path) for name, path in files.items()}
    module.stage_inputs(files, {'state.nc'}, out, ingress)
    assert not (out / 'state.nc').is_symlink()
    assert (out / 'state.nc').stat().st_mode & 0o200
    (out / 'state.nc').write_bytes(b'native-rewritten-state')
    assert source.read_bytes() == b'ingress-bytes' and module.sha(source) == ingress['state.nc']
    assert (out / 'table.TBL').is_symlink()
    with pytest.raises(ValueError, match='input/deck collision'):
        module.stage_inputs(files, {'state.nc'}, out, ingress)
    changed = tmp_path / 'changed-run'; changed.mkdir()
    with pytest.raises(ValueError, match='changed during input copy'):
        module.stage_inputs(files, {'state.nc'}, changed, dict(ingress, **{'state.nc':'incorrect-ingress-hash'}))


def test_workflow_overlay_authenticates_source_and_preserves_all_stream_lists(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_workflow_overlay', ROOT / 'workflow_deck.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ref = Path(os.environ.get('MPAS_LANES', '.')) / 'ref/rrfs'
    original = ref / 'decks/gsl-ci-source-v1/ufscommunity.hrrrv5.gfs.winter'
    if not original.exists():
        pytest.skip('external immutable GSL deck absent')
    request = dict(stream_list_directory='fix/stream_list/hrrrv5',
        settings=dict(config_jedi_da=True, config_do_restart=True,
                      config_start_time='2023-03-31_13:00:00', config_run_duration='0_01:00:00',
                      config_dt=60.0, config_number_of_sub_steps=8),
        stream_variables=dict(mpasout_interval='01:00:00', restart_interval='01:00:00',
            history_interval='01:00:00', diag_interval='01:00:00', lbc_interval=3),
        streams=dict(da_state=dict(output_interval=None, output_timelevels='0 1')))
    deck = module.generate(original, ref / 'rrfs-workflow', ref / 'MPAS-Model-rrfs', tmp_path / 'deck', request)
    index = json.loads((deck.parent / 'INDEX.json').read_text())
    assert index['decks'][0]['resolved_settings']['config_do_restart'] is True
    assert index['runtime']['start'] == request['settings']['config_start_time']
    assert index['runtime']['duration'] == request['settings']['config_run_duration']
    assert index['runtime']['dt'] == request['settings']['config_dt']
    assert index['runtime']['substeps'] == request['settings']['config_number_of_sub_steps']
    assert not ET.parse(deck / 'streams.atmosphere').find("stream[@name='da_state']").get('output_interval')
    for name, expected in index['decks'][0]['files'].items():
        assert module.generator.f0.sha(deck / name) == expected
    copied = tmp_path / 'corrupt'; shutil.copytree(original.parent, copied)
    changed = copied / original.name / 'namelist.atmosphere'; changed.chmod(0o644); changed.write_text('changed')
    with pytest.raises(ValueError, match='immutable index'):
        module.generate(copied / original.name, ref / 'rrfs-workflow', ref / 'MPAS-Model-rrfs', tmp_path / 'refused', request)


def test_thompson_table_runner_refuses_unaudited_or_replaced_utility_before_execution(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_native_table_guard', ROOT / 'run_tables.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    build = tmp_path / 'build'; build.mkdir()
    audit = dict(all_ok=True, core='atmosphere', recipe='gnu-strict', objects=[])
    audit_file = build / 'flag-audit.json'; audit_file.write_text(json.dumps(audit))
    out = tmp_path / 'tables'
    with pytest.raises(ValueError, match='object not audited'):
        module.run_tables(build, out, tmp_path / 'mpi', 1, None, None, 'thompson')
    assert not out.exists()
    audit['objects'] = [dict(object='src/core_atmosphere/utils/' + name, flags_ok=True)
                        for name in ('build_tables.o', 'atmphys_build_tables_thompson.o')]
    audit_file.write_text(json.dumps(audit))
    (build / 'MANIFEST.json').write_text(json.dumps(dict(
        flags_sha256=module.sha(audit_file), utility_sha256={'build_tables': '0' * 64})))
    (build / 'build_tables').write_bytes(b'replaced utility')
    with pytest.raises(ValueError, match='utility differs'):
        module.run_tables(build, out, tmp_path / 'mpi', 1, None, None, 'thompson')
    assert not out.exists()


def test_recorded_output_comparison_refuses_corruption_before_invoking_rust(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_recorded_outputs', ROOT / 'identity.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    runs = [tmp_path / 'left', tmp_path / 'right']
    manifests = []
    for run in runs:
        run.mkdir(); (run / 'changed-da.nc').write_bytes(b'native-output')
        (run / 'unchanged-init.nc').write_bytes(b'copied-ingress')
        manifests.append(dict(outputs={'changed-da.nc':module.digest(run / 'changed-da.nc')}))
    (runs[1] / 'changed-da.nc').write_bytes(b'corrupt-transfer')
    with pytest.raises(ValueError, match='stream output differs'):
        module.compare_outputs(runs, manifests, tmp_path / 'rust-must-not-run')


def test_complete_io_replay_rejects_changed_runtime_before_native_execution(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_complete_io_guard', ROOT / 'replay_io.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    table = tmp_path / 'native-table'; table.write_bytes(b'changed-table')
    request = dict(mpi=str(tmp_path / 'native-must-not-run'),
        table_hashes={str(table):'original-table-digest'})
    with pytest.raises(ValueError, match='runtime table bytes changed'):
        module.replay(request, tmp_path / 'whole-attempt')
    assert json.loads((tmp_path / 'whole-attempt/REQUEST.json').read_text()) == request
    assert not (tmp_path / 'whole-attempt/SUMMARY.json').exists()


def test_received_forecast_authenticates_build_and_unchanged_source_witness(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_received_forecast_guard', ROOT / 'receive_forecast.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    build, source, ref = [tmp_path / name for name in ('build', 'source', 'ref')]
    for path in (build, source, ref):path.mkdir()
    sha = module.publisher.sha
    (build / 'atmosphere_model').write_bytes(b'audited-binary')
    (build / 'toolchain.txt').write_bytes(b'toolchain')
    (build / 'SOURCE-PINS.json').write_text(json.dumps(dict(fork='primary')))
    (build / 'flag-audit.json').write_text(json.dumps(dict(all_ok=True, core='atmosphere', recipe='gnu-strict')))
    native = dict(pin_set_hash='pins', binary_sha256=sha(build / 'atmosphere_model'),
        flags_sha256=sha(build / 'flag-audit.json'),toolchain_sha256=sha(build / 'toolchain.txt'))
    (build / 'MANIFEST.json').write_text(json.dumps(native))
    (source / 'state.nc').write_bytes(b'emitted-native-bytes')
    (source / 'namelist.atmosphere').write_bytes(b'deck')
    m = dict(completed=True, exit_status=0, recipe='gnu-strict', fork='primary', ranks=1, deck='deck',runtime={},
        pin_set_hash='pins',binary_sha256=native['binary_sha256'],toolchain_sha256=native['toolchain_sha256'],
        source_inputs_unchanged=False, outputs={'state.nc':sha(source / 'state.nc')},
        resolved_configuration=dict(files={'namelist.atmosphere':sha(source / 'namelist.atmosphere')}))
    def expected():
        (source / 'MANIFEST.json').write_text(json.dumps(m))
        return dict(manifest_sha256=sha(source / 'MANIFEST.json'),mesh='mesh',date='date',label='run')
    with pytest.raises(ValueError,match='unchanged ingress'):
        module.receive(source,ref,build,expected())
    m['source_inputs_unchanged']=True
    (build / 'atmosphere_model').write_bytes(b'changed-binary')
    with pytest.raises(ValueError,match='pinned audited'):
        module.receive(source,ref,build,expected())
    (build / 'atmosphere_model').write_bytes(b'audited-binary')
    (source / 'unexpected.nc').write_bytes(b'private-input')
    with pytest.raises(ValueError,match='inventory differs'):
        module.receive(source,ref,build,expected())
    (source / 'unexpected.nc').unlink()
    assert Path(module.receive(source,ref,build,expected())['path']).joinpath('state.nc').is_file()


def test_s3_etag_replaces_contaminated_input_before_recording_it(tmp_path, monkeypatch):
    import hashlib
    spec = importlib.util.spec_from_file_location('rrfs_verified_download', ROOT / 'download_verified_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b'public-reference-bytes'
    etag = hashlib.md5(payload).hexdigest()
    row = dict(name='input.grib2', url='https://public.example/input.grib2')
    (tmp_path / row['name']).write_bytes(payload + b'concurrent-append')
    def headers(argv, check):
        Path(argv[argv.index('--output')+1]).write_text(
            f'HTTP/1.1 200 OK\nContent-Length: {len(payload)}\nETag: "{etag}"\n')
    def download(request, directory):
        (directory / request['name']).write_bytes(payload)
        return dict(request, bytes=len(payload))
    monkeypatch.setattr(module.subprocess, 'run', headers)
    monkeypatch.setattr(module, 'download', download)
    result = module.verified(row, tmp_path)
    assert result['verified'] and result['replaced_invalid_input']['bytes'] > len(payload)
    assert result['sha256'] == hashlib.sha256(payload).hexdigest()
    assert json.loads((tmp_path / 'input.grib2.VERIFIED.json').read_text())['s3_etag'] == etag


def test_decode_reuse_binds_bytes_after_storage_relocation(tmp_path):
    import hashlib
    spec = importlib.util.spec_from_file_location('rrfs_matrix_run', ROOT / 'run_case_matrix.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoded = tmp_path / 'decoded'; decoded.mkdir()
    sources = [tmp_path / 'main.grib2', tmp_path / 'extra.grib2']
    for p in sources:p.write_bytes(p.name.encode())
    binary, table = tmp_path / 'ungrib', tmp_path / 'Vtable'
    binary.write_bytes(b'native');table.write_bytes(b'variables')
    intermediate = decoded / 'GFS:2023-03-31_12'; intermediate.write_bytes(b'native-weather')
    row = dict(start='2023-03-31_12:00:00',stop='2023-03-31_12:00:00',interval_seconds=3600,
               gribs=[[str(p) for p in sources]])
    sha = module.cache.sha
    receipt = dict(row=row, inputs={str(Path('/old/storage') / p.name):sha(p) for p in sources},
                   ungrib_sha256=sha(binary),vtable_sha256=sha(table),
                   intermediates={intermediate.name:sha(intermediate)})
    (decoded / 'MANIFEST.json').write_text(json.dumps(receipt))
    assert module.verified_decode(row,decoded,binary,table) == decoded
    sources[0].write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs from requested inputs'):
        module.verified_decode(row,decoded,binary,table)


def test_forecast_publication_checks_recorded_bytes_and_retains_only_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_forecast_publication', ROOT / 'publish_forecast.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'run'; source.mkdir()
    output = source / 'history.nc';output.write_bytes(b'actual-native-output')
    deck = source / 'namelist.atmosphere';deck.write_bytes(b'native deck')
    (source / 'init.nc').write_bytes(b'input-must-stay-in-case-cache')
    reference = tmp_path / 'reference';reference.mkdir()
    manifest = dict(completed=True, exit_status=0, pin_set_hash='pins',recipe='gnu-strict',deck='deck',
                    ranks=1,runtime=dict(duration='24h'),outputs={'history.nc':module.sha(output)},
                    resolved_configuration=dict(files={'namelist.atmosphere':module.sha(deck)}))
    (source / 'MANIFEST.json').write_text(json.dumps(manifest))
    output.write_bytes(b'changed-after-run')
    with pytest.raises(ValueError,match='differs from native receipt'):
        module.publish_forecast(source,reference,'mesh','date','run1')
    output.write_bytes(b'actual-native-output')
    result = module.publish_forecast(source,reference,'mesh','date','run1')
    published = Path(result['path'])
    assert (published / 'history.nc').read_bytes() == output.read_bytes()
    assert not (published / 'init.nc').exists()


def test_case_reuse_and_publication_bind_inputs_outputs_and_workflow_deck(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_reuse', ROOT / 'cache_cases.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    workflow, build, decoded, out = [tmp_path / name for name in ('workflow', 'build', 'decoded', 'run')]
    for path in (workflow, build, decoded, out): path.mkdir()
    (workflow / 'namelist').write_text('&init\n config_init_case = ${case}\n/\n')
    (workflow / 'streams').write_text('<streams><stream name="input" filename_template="@input@"/></streams>')
    row = dict(variables=dict(case=7), ranks=1, namelist_template='namelist', streams_template='streams',
               stream_replacements=dict(input='static.nc'), expected_outputs=['init.nc'],
               inputs={'static.nc':str(tmp_path / 'static.nc')}, decoded=str(decoded))
    (tmp_path / 'static.nc').write_bytes(b'input')
    (build / 'init_atmosphere_model').write_bytes(b'native')
    (build / 'flag-audit.json').write_text('{}')
    weather = decoded / 'GFS:2023-03-31_12'; weather.write_bytes(b'decoded-weather')
    (decoded / 'MANIFEST.json').write_text(json.dumps(dict(intermediates={weather.name:module.sha(weather)})))
    for name, content in module.configuration(row,workflow).items(): (out / name).write_text(content)
    (out / 'init.nc').write_bytes(b'native-initialized-weather')
    (out / 'static.nc').symlink_to(tmp_path / 'static.nc')
    receipt = dict(completed=True,exit_status=0,row=row,binary_sha256=module.sha(build / 'init_atmosphere_model'),
                   audit_sha256=module.sha(build / 'flag-audit.json'),
                   workflow_templates={name:module.sha(workflow / name) for name in ('namelist','streams')},
                   inputs={'static.nc':module.sha(tmp_path / 'static.nc')},runtime_files={},
                   decoded_manifest_sha256=module.sha(decoded / 'MANIFEST.json'),
                   outputs={'init.nc':module.sha(out / 'init.nc')},
                   configuration={name:module.sha(out / name) for name in module.configuration(row,workflow)})
    (out / 'MANIFEST.json').write_text(json.dumps(receipt))
    assert module.verified_case(row,out,workflow,build,tmp_path / 'unused-mpi') == receipt
    ready = tmp_path / 'READY.md'; destination = tmp_path / 'published'
    publication = module.publish_case(out,destination,ready,'case')
    assert not (destination / 'static.nc').exists()
    assert module.publish_case(out,destination,ready,'case') == publication
    (out / 'namelist.init_atmosphere').write_text('changed')
    with pytest.raises(ValueError,match='deck bytes changed'):
        module.verified_case(row,out,workflow,build,tmp_path / 'unused-mpi')
    with pytest.raises(ValueError,match='configuration differs'):
        module.publish_case(out,tmp_path / 'changed-publication',ready,'case')
    (out / 'namelist.init_atmosphere').write_text(module.configuration(row,workflow)['namelist.init_atmosphere'])
    weather.write_bytes(b'changed-decoded-weather')
    with pytest.raises(ValueError,match='intermediate bytes changed'):
        module.verified_case(row,out,workflow,build,tmp_path / 'unused-mpi')


@pytest.mark.parametrize('kind', ['init', 'gfl'])
def test_pins_separate_approved_authorities_from_pristine(tmp_path, kind):
    spec = importlib.util.spec_from_file_location('rrfs_pins_authority', ROOT / 'record_pins.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location('rrfs_ruled_pins', ROOT / 'ruled_reference.py')
    ruled = importlib.util.module_from_spec(spec); spec.loader.exec_module(ruled)
    reference = tmp_path / 'ref'
    directory = reference / 'native' / 'derivatives' / kind
    directory.mkdir(parents=True)
    digest = ruled.INIT_BINARY if kind == 'init' else 'a' * 64
    manifest = dict(schema='mpas.rrfs-native-build/v1', reference=kind,
                    recipe='gnu-strict', core='init_atmosphere' if kind == 'init' else 'atmosphere',
                    binary_sha256=digest, toolchain_sha256='b' * 64, flags_sha256='c' * 64,
                    source_correction=ruled.authority(kind, digest))
    def seal():
        (directory / 'MANIFEST.json').write_text(json.dumps(manifest))
        (directory / 'PUBLICATION.json').write_text(json.dumps(dict(files={
            'MANIFEST.json':module.sha(directory / 'MANIFEST.json')})))
    seal()
    document = tmp_path / 'PINS.json'; document.write_text('{}')
    result = module.record(reference, document)
    assert not result['native_builds']
    assert len(result['source_corrected_builds']) == 1
    manifest['source_correction']['label'] = 'pristine'
    seal()
    with pytest.raises(ValueError, match='binding differs'):
        module.record(reference, document)


@pytest.mark.parametrize('changed', ["config_smoke_scheme = 'on'", "config_dust_scheme = 'on'",
                                    "config_smoke_scheme = 'unknown'", "config_smoke_scheme = 'off'\nconfig_smoke_scheme = 'off'",
                                    "config_smoke_scheme = 'OFF'", "config_smoke_scheme = 'off', config_dust_scheme = 'on'"])
def test_approved_initializer_cannot_generate_noncanonical_aerosol_inputs(changed):
    spec = importlib.util.spec_from_file_location('rrfs_ruled_canonical', ROOT / 'ruled_reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    valid = "config_smoke_scheme = 'off'\nconfig_dust_scheme = 'off'\n"
    module.validate_init_configuration(valid)
    module.validate_init_configuration('')  # Both pinned registry defaults are off.
    module.validate_init_configuration("! config_smoke_scheme = 'on'\n" + valid)
    text = valid.replace("config_smoke_scheme = 'off'", changed) if not changed.startswith('config_dust_scheme') else valid.replace("config_dust_scheme = 'off'", changed)
    with pytest.raises(ValueError, match='canonical aerosol-off configuration'):
        module.validate_init_configuration(text)


def test_diagnostic_build_cannot_become_the_reference(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_publish_build', ROOT / 'publish_build.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path / 'flag-audit.json').write_text(json.dumps(dict(all_ok=True,core='init_atmosphere',recipe='gnu-strict')))
    (tmp_path / 'SOURCE-PINS.json').write_text(json.dumps(dict(diagnostic='bounds-snan')))
    with pytest.raises(ValueError,match='diagnostic binaries'):
        module.publish_build(tmp_path,tmp_path / 'ref','init_atmosphere','gnu-strict')


def test_diagnostic_initialized_weather_cannot_become_primary(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_diagnostic_guard', ROOT / 'cache_cases.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'run'; source.mkdir()
    (source / 'MANIFEST.json').write_text(json.dumps(dict(completed=True, exit_status=0,
        source_pins=dict(diagnostic='counterfactual-optional-aerosol-nullify'))))
    with pytest.raises(ValueError, match='diagnostic initializer outputs'):
        module.publish_case(source, tmp_path / 'primary', tmp_path / 'READY.md', 'case')
    assert not (tmp_path / 'primary').exists()
    assert not (tmp_path / 'READY.md').exists()


def test_received_case_rejects_changed_geography_and_initializer(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_case_receiver_guard', ROOT / 'receive_case.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, build = [tmp_path / name for name in ('source', 'build')]
    for path in (source, build):path.mkdir()
    sha = module.cache.sha
    (build / 'init_atmosphere_model').write_bytes(b'pristine-native')
    pins = dict(fork='primary');(build / 'SOURCE-PINS.json').write_text(json.dumps(pins))
    (build / 'flag-audit.json').write_text(json.dumps(dict(all_ok=True, recipe='gnu-strict',core='init_atmosphere')))
    (source / 'static.nc').write_bytes(b'native-static')
    (source / 'namelist.init_atmosphere').write_bytes(b'workflow-static')
    geography = source / 'GEOGRAPHY-MANIFEST.json';geography.write_bytes(b'pinned-geography-index')
    m = dict(source_pins=pins,completed=True,exit_status=0, binary_sha256=sha(build / 'init_atmosphere_model'),
        audit_sha256=sha(build / 'flag-audit.json'),row=dict(geography_source_manifest_sha256=sha(geography)),
        outputs={'static.nc':sha(source / 'static.nc')},configuration={'namelist.init_atmosphere':sha(source / 'namelist.init_atmosphere')})
    (source / 'MANIFEST.json').write_text(json.dumps(m))
    expected = dict(manifest_sha256=sha(source / 'MANIFEST.json'),description='pristine static')
    geography.write_bytes(b'changed-geography-index')
    with pytest.raises(ValueError,match='geography source manifest differs'):
        module.receive(source,tmp_path / 'refused',tmp_path / 'READY.md',build,expected)
    geography.write_bytes(b'pinned-geography-index')
    (build / 'init_atmosphere_model').write_bytes(b'changed-native')
    with pytest.raises(ValueError,match='audited strict initializer authority'):
        module.receive(source,tmp_path / 'refused',tmp_path / 'READY.md',build,expected)
    (build / 'init_atmosphere_model').write_bytes(b'pristine-native')
    publication=module.receive(source,tmp_path / 'published',tmp_path / 'READY.md',build,expected)
    assert (Path(publication['path']) / 'GEOGRAPHY-MANIFEST.json').read_bytes() == geography.read_bytes()


@pytest.mark.parametrize('kind,table', [('tempo', 'freeze_water_data_tempo_v3'),
                                      ('thompson', 'MP_THOMPSON_QRacrQG_DATA.DBL')])
def test_pins_include_verified_tables_in_the_published_table_cache(tmp_path, kind, table):
    spec = importlib.util.spec_from_file_location('rrfs_pins_table_cache', ROOT / 'record_pins.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    reference = tmp_path / 'ref'; tables = reference / 'tables' / 'generated'; tables.mkdir(parents=True)
    (tables / table).write_bytes(b'native table')
    (tables / 'MANIFEST.json').write_text(json.dumps(dict(schema=f'mpas.rrfs-{kind}-native-tables/v1',
        completed=True, outputs={table:module.sha(tables / table)})))
    (tables / 'PUBLICATION.json').write_text(json.dumps(dict(files={name:module.sha(tables / name)
        for name in ['MANIFEST.json', table]})))
    document = tmp_path / 'PINS.json'; document.write_text(json.dumps(dict(other_reference='preserved')))
    result = module.record(reference, document)
    assert result['native_tables']['generated']['table_sha256'][table] == module.sha(tables / table)
    assert json.loads(document.read_text())['other_reference'] == 'preserved'
    (tables / table).write_bytes(b'changed table')
    with pytest.raises(ValueError, match='published RRFS product changed'):
        module.record(reference, document)
    (tables / table).write_bytes(b'native table')
    publication = json.loads((tables / 'PUBLICATION.json').read_text())
    del publication['files'][table]
    (tables / 'PUBLICATION.json').write_text(json.dumps(publication))
    with pytest.raises(ValueError, match='outputs differ from publication authority'):
        module.record(reference, document)


def test_raw_input_publication_checks_copied_bytes_before_ready(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_input_publication', ROOT / 'publish_inputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'raw'; source.mkdir()
    data = source / 'input.grib2'; data.write_bytes(b'verified-public-bytes')
    row = dict(name=data.name,verified=True,bytes=data.stat().st_size,sha256=module.sha(data))
    (source / (data.name+'.VERIFIED.json')).write_text(json.dumps(row))
    (source / 'MANIFEST.json').write_text(json.dumps(dict(completed=True,files=[row])))
    ready = tmp_path / 'READY.md'
    result = module.publish_inputs(source,tmp_path / 'published',ready,tmp_path / 'staging-ready')
    assert result['publication_sha256'] in ready.read_text()
    data.write_bytes(b'changed-after-download')
    with pytest.raises(ValueError,match='differs from verified S3 bytes'):
        module.publish_inputs(source,tmp_path / 'invalid-publication',ready,tmp_path / 'staging-ready')
    assert not (tmp_path / 'invalid-publication').exists()
    assert 'invalid-publication' not in ready.read_text()


def test_native_input_transport_rejects_corruption_before_exposing_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_transport', ROOT / 'transport_inputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'published-input'; source.write_bytes(bytes(range(256)) * 3)
    expected = module.sha(source)
    with pytest.raises(ValueError, match='source differs'):
        module.pack({str(source): '0'*64},tmp_path/'bad-source',chunk_bytes=100)
    manifest = module.pack({str(source): expected},tmp_path/'packed',chunk_bytes=100)
    assert len(manifest['files'][str(source)]['chunks']) == 8
    first = tmp_path/'packed'/manifest['files'][str(source)]['chunks'][0]['file']
    original = first.read_bytes();first.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='chunk differs'):
        module.unpack(tmp_path/'packed',tmp_path/'restored')
    assert not (tmp_path/'restored'/expected).exists()
    first.write_bytes(original)
    mapping = module.unpack(tmp_path/'packed',tmp_path/'restored')
    assert Path(mapping[str(source)]).read_bytes() == source.read_bytes()
    assert module.unpack(tmp_path/'packed',tmp_path/'restored') == mapping


def test_case_namelist_companion_is_bound_to_source_templates(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_companion', ROOT / 'cache_cases.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'main').write_text('&init\n config_init_case = ${case}\n/\n')
    (tmp_path/'chem').write_text('&chem\n enabled = .true.\n/\n')
    (tmp_path/'streams').write_text('<streams/>')
    row=dict(variables=dict(case=7),namelist_template='main',streams_template='streams',
             namelist_companion_templates=['chem'],stream_replacements={},text_replacements=[])
    assert '&chem' in module.configuration(row,tmp_path)['namelist.init_atmosphere']
    expected = module.template_hashes(row,tmp_path)
    (tmp_path/'chem').write_text('&chem\n enabled = .false.\n/\n')
    assert module.template_hashes(row,tmp_path)['chem'] != expected['chem']


def test_tempo_cloud_companion_applies_only_to_its_native_scheme():
    module=load_decks();rules=json.loads((ROOT/'deck-specs.json').read_text())['companion_rules']
    assert module.companions(dict(config_microp_scheme='mp_tempo',config_tempo_cldfra=True),rules)['config_radt_cld_scheme']=='cld_incidence'
    assert 'config_radt_cld_scheme' not in module.companions(dict(config_microp_scheme='off',config_tempo_cldfra=True),rules)


def test_forecast_matrix_preserves_pending_inputs_without_claiming_success(tmp_path):
    spec=importlib.util.spec_from_file_location('rrfs_forecast_matrix',ROOT/'run_forecast_matrix.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=dict(name='required-run',build_receipt=str(tmp_path/'missing-native-build.json'))
    request=dict(rows=[row]);receipt=tmp_path/'attempt.json'
    assert not module.execute(request,tmp_path/'work',tmp_path/'reference',receipt,once=True)
    result=json.loads(receipt.read_text())
    assert not result['completed'] and result['results']['required-run']['state']=='pending'
    assert not (tmp_path/'work').exists()
    changed=dict(rows=[dict(row,name='different-contract')])
    with pytest.raises(ValueError,match='different request'):
        module.execute(changed,tmp_path/'work',tmp_path/'reference',receipt,once=True)
    with pytest.raises(ValueError,match='simple unique'):
        module.execute(dict(rows=[dict(row,name='../unsafe')]),tmp_path/'work',tmp_path/'reference',tmp_path/'unsafe.json',once=True)


def test_whole_forecast_replay_refuses_unpinned_inputs_and_future_restart_before_start(tmp_path):
    spec = importlib.util.spec_from_file_location('rrfs_whole_replay', ROOT/'replay_matrix.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    native = tmp_path/'original.nc'; native.write_bytes(b'original native input')
    request = dict(work_root=str(tmp_path), runs=[dict(name='cold')],
                   input_hashes={str(native):module.load('run_forecast').sha(native)})
    native.write_bytes(b'changed input bytes')
    with pytest.raises(ValueError, match='input authority changed'):
        module.matrix(request, tmp_path/'changed-input-attempt')
    assert not (tmp_path/'changed-input-attempt').exists()
    request['runs'] = [dict(name='warm', from_outputs={'cold':{}}), dict(name='cold')]
    with pytest.raises(ValueError, match='dependency must precede'):
        module.matrix(request, tmp_path/'future-restart-attempt')
    assert not (tmp_path/'future-restart-attempt').exists()
    request['runs'] = [dict(name='..')]
    with pytest.raises(ValueError, match='simple forecast identifiers'):
        module.matrix(request, tmp_path/'unsafe-attempt')
    assert not (tmp_path/'unsafe-attempt').exists()


def test_hourly_decode_refuses_changed_transfer_before_native_decode(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('rrfs_hourly_decode',ROOT/'decode_hourly.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    binary=tmp_path/'ungrib';binary.write_bytes(b'pinned binary')
    table=tmp_path/'Vtable';table.write_bytes(b'pinned workflow table')
    request=dict(source=dict(host='source',directory='/immutable',rsync_argv=['rsync']),
        ungrib_sha256=module.sha(binary),vtable_sha256=module.sha(table),
        raw_source_publication_sha256='source-pin',interval_seconds=3600,
        rows=[dict(name='hour',valid_time='2000-01-01_00:00:00',inputs=[dict(name='input',bytes=5,sha256='0'*64)])])
    def changed_transfer(argv,check):Path(argv[-1]).write_bytes(b'wrong')
    def native_must_not_run(*args):raise AssertionError('corrupted weather bytes reached native decode')
    monkeypatch.setattr(module.subprocess,'run',changed_transfer)
    monkeypatch.setattr(module,'decode',native_must_not_run)
    with pytest.raises(ValueError,match='received raw input bytes differ'):
        module.execute(request,tmp_path/'work',binary,table)
    assert not (tmp_path/'work/hour').exists()


def test_dry_measurement_retains_interval_diagnostics_but_rejects_io_failure():
    spec=importlib.util.spec_from_file_location('rrfs_dry_completion',ROOT/'measure_dycore.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text="ERROR: Stream 'diagnostics' does not exist. Cannot retrieve stream input/output intervals\nFinished running the atmosphere core\n"
    result=module.completion_witness(text)
    assert result['completed'] and result['missing_diagnostics_interval_messages']==1
    result=module.completion_witness(text+'ERROR SMIOL write failed: no space left\n')
    assert not result['completed'] and len(result['other_error_messages'])==1
    assert not module.completion_witness(text.replace('Finished running the atmosphere core','aborted'))['completed']


def test_dry_measurement_recovery_preserves_failed_wrapper_and_requires_retired_hash(tmp_path):
    spec=importlib.util.spec_from_file_location('rrfs_dry_recovery',ROOT/'measure_dycore.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    build=tmp_path/'build';build.mkdir()
    for name,data in [('atmosphere_model',b'strict executable'),('toolchain.txt',b'pinned compiler')]:
        (build/name).write_bytes(data)
    (build/'flag-audit.json').write_text(json.dumps(dict(all_ok=True,recipe='gnu-strict')))
    binary=module.digest(build/'atmosphere_model')
    (build/'MANIFEST.json').write_text(json.dumps(dict(core='atmosphere',recipe='gnu-strict',binary_sha256=binary,pin_set=dict(fork='fork-pin'))))
    run=tmp_path/'primary';run.mkdir();(run/'first.nc').write_bytes(b'original weather words')
    (run/'log.atmosphere.0000.out').write_text("ERROR: Stream 'diagnostics' does not exist. Cannot retrieve stream input/output intervals\nFinished running the atmosphere core\n")
    original=dict(build=str(build),source_pin='fork-pin',binary_sha256=binary,
        audit_sha256=module.digest(build/'flag-audit.json'),toolchain_sha256=module.digest(build/'toolchain.txt'),
        outputs={'first.nc':module.digest(run/'first.nc'),'retired.nc':'1'*64},exit_status=0,completed=False)
    (run/'MANIFEST.json').write_text(json.dumps(original));original_hash=module.digest(run/'MANIFEST.json')
    retired=tmp_path/'retired.json';retired.write_text(json.dumps(dict(retired={})))
    with pytest.raises(ValueError,match='missing output lacks matching retired-byte witness'):
        module.recover_measurement(run,tmp_path/'invalid',retired,['first.nc'])
    assert not (tmp_path/'invalid').exists()
    retired.write_text(json.dumps(dict(retired={'primary/retired.nc':dict(sha256='1'*64)})))
    result=module.recover_measurement(run,tmp_path/'recovered',retired,['first.nc'])
    assert result['completed'] and result['measurement_only']
    assert module.digest(run/'MANIFEST.json')==original_hash
    assert json.loads((tmp_path/'recovered/ORIGINAL-MANIFEST.json').read_text())['completed'] is False
    (run/'first.nc').write_bytes(b'tampered weather words')
    with pytest.raises(ValueError,match='native output changed'):
        module.recover_measurement(run,tmp_path/'tampered',retired,['first.nc'])


def test_case_without_weather_inputs_requires_explicit_disabled_stages():
    spec=importlib.util.spec_from_file_location('rrfs_static_inputs',ROOT/'cache_cases.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text='&preproc_stages\n config_met_interp = false\n config_input_sst = .false.\n/\n&vertical_grid\n config_blend_bdy_terrain = false\n/\n'
    assert module.decoded_directory({}, {'namelist.init_atmosphere':text}) is None
    for changed in [text.replace('config_met_interp = false','config_met_interp = true'),
                    text.replace('config_input_sst = .false.',''),
                    text+'config_met_interp = false\n',
                    text.replace('config_blend_bdy_terrain = false', 'config_blend_bdy_terrain = true'),
                    text.replace('config_blend_bdy_terrain = false', '')]:
        with pytest.raises(ValueError,match='missing decoded inputs'):
            module.decoded_directory({}, {'namelist.init_atmosphere':changed})


def test_geography_archive_guards_before_native_extraction(tmp_path):
    import io
    import shutil
    import tarfile
    spec=importlib.util.spec_from_file_location('rrfs_geography',ROOT/'stage_geography.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tar=Path(shutil.which('tar')).resolve()
    archive=tmp_path/'geography.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo('dataset/index');member.size=4;stream.addfile(member,io.BytesIO(b'grid'))
    row=dict(path=str(archive),bytes=archive.stat().st_size,sha256=module.sha(archive))
    index=module.preflight(dict(archives=[row]),tar)
    with pytest.raises(ValueError,match='storage bound'):
        module.extract(index,tar,tmp_path/'too-large',tmp_path,1)
    assert not (tmp_path/'too-large').exists()
    result=module.extract(index,tar,tmp_path/'geography',tmp_path,1<<20)
    assert result['indexes']['dataset/index']==module.sha(tmp_path/'geography/dataset/index')
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo('../escape');member.size=4;stream.addfile(member,io.BytesIO(b'grid'))
    with pytest.raises(ValueError,match='pinned bytes'):
        module.preflight(dict(archives=[row]),tar)
    row.update(bytes=archive.stat().st_size,sha256=module.sha(archive))
    with pytest.raises(ValueError,match='unsafe geography archive member'):
        module.preflight(dict(archives=[row]),tar)


def test_sparse_geography_preserves_bytes_and_bounds_each_member(tmp_path):
    import io
    import shutil
    import subprocess
    import tarfile
    spec=importlib.util.spec_from_file_location('rrfs_sparse_geography',ROOT/'stage_geography.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tar=Path(shutil.which('tar')).resolve()
    data=b'original-prefix'+bytes(8<<20)+b'original-suffix'
    archive=tmp_path/'sparse.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo('dataset/field');member.size=len(data)
        stream.addfile(member,io.BytesIO(data))
    row=dict(path=str(archive),bytes=archive.stat().st_size,sha256=module.sha(archive))
    index=module.preflight(dict(archives=[row]),tar)
    result=module.extract(index,tar,tmp_path/'sparse',tmp_path,32<<20,sparse=True,reserve_bytes=1<<20)
    field=tmp_path/'sparse/dataset/field'
    assert field.read_bytes()==data
    assert field.stat().st_size==len(data)
    assert field.stat().st_blocks*512<len(data)//2
    assert result['allocated_file_bytes']==field.stat().st_blocks*512
    assert result['cp_sha256']==module.sha(Path(shutil.which('cp')).resolve())
    with pytest.raises(subprocess.CalledProcessError):
        module.extract(index,tar,tmp_path/'bounded',tmp_path,4<<20,sparse=True,reserve_bytes=1<<20)
    assert not (tmp_path/'bounded/dataset/field').exists()


def test_compressed_geography_checks_source_and_owned_paths_before_mount(tmp_path, monkeypatch):
    import hashlib
    import io
    import shutil
    import tarfile
    spec = importlib.util.spec_from_file_location('rrfs_compressed_geography', ROOT/'stage_geography.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    work = tmp_path/'work'; work.mkdir()
    temporary = tmp_path/'temporary'; temporary.mkdir()
    monkeypatch.setenv('TMPDIR', str(temporary))
    archive = work/'geography.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo('dataset/index'); member.size = 4
        stream.addfile(member, io.BytesIO(b'grid'))
    tar = Path(shutil.which('tar')).resolve()
    index = module.preflight(dict(archives=[dict(path=str(archive), bytes=archive.stat().st_size,
                                               sha256=module.sha(archive))]), tar)
    absent = work/'uninstalled-native-tool'
    pinned = hashlib.sha256((json.dumps(index, indent=2)+'\n').encode()).hexdigest()
    with monkeypatch.context() as authenticated:
        authenticated.setattr(module, 'preflight', lambda *args: pytest.fail('pinned bytes were redundantly listed'))
        with pytest.raises(ValueError, match='tools are absent'):
            with module.compressed_view(index, tar, work/'authenticated', work,
                                        temporary/'view', 1<<20, absent, absent, work,
                                        preflight_index_sha256=pinned):
                pytest.fail('absent filesystem tool reached native mount')
        with pytest.raises(ValueError, match='preflight receipt changed'):
            with module.compressed_view(dict(index, extracted_bytes=1), tar,
                                        work/'changed-receipt', work, temporary/'view',
                                        1<<20, absent, absent, work,
                                        preflight_index_sha256=pinned):
                pytest.fail('changed preflight receipt reached native mount')
    with pytest.raises(ValueError, match='owned work and TMPDIR'):
        with module.compressed_view(index, tar, tmp_path/'outside-work', work,
                                    temporary/'view', 1<<20, absent, absent, work):
            pytest.fail('escaped work root was mounted')
    with pytest.raises(ValueError, match='storage bound'):
        with module.compressed_view(index, tar, work/'bounded', work,
                                    temporary/'view', 1, absent, absent, work):
            pytest.fail('storage refusal reached native mount')
    archive.write_bytes(archive.read_bytes() + b'changed-source')
    with pytest.raises(ValueError, match='pinned bytes'):
        with module.compressed_view(index, tar, work/'changed-pinned', work,
                                    temporary/'view', 1<<20, absent, absent, work,
                                    preflight_index_sha256=pinned):
            pytest.fail('pinned receipt bypassed changed archive bytes')
    with pytest.raises(ValueError, match='pinned bytes'):
        with module.compressed_view(index, tar, work/'changed', work,
                                    temporary/'view', 1<<20, absent, absent, work):
            pytest.fail('changed source was mounted')
    assert not (temporary/'view').exists()
    assert not (work/'changed').exists()


def test_reused_geography_rejects_changed_receipt_source_and_images(tmp_path,monkeypatch):
    import hashlib
    spec=importlib.util.spec_from_file_location('rrfs_geography_reuse',ROOT/'stage_geography.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    work=tmp_path/'work';work.mkdir();temporary=tmp_path/'temporary';temporary.mkdir()
    monkeypatch.setenv('TMPDIR',str(temporary))
    images=work/'images';images.mkdir();archive=work/'source.tar.gz';archive.write_bytes(b'original archive')
    image=images/'original.sqfs';image.write_bytes(b'original image')
    tool=work/'mount-tool';tool.write_bytes(b'never executed in negative controls')
    receipt=dict(completed=True,storage='native-read-only-compressed-filesystem',
                 squashfuse_sha256=module.sha(tool),indexes={},
                 archives=[dict(path=str(archive),bytes=archive.stat().st_size,sha256=module.sha(archive),roots=['dataset'])],
                 products=[dict(source_sha256=module.sha(archive),image=image.name,bytes=image.stat().st_size,
                                image_sha256=module.sha(image),all_regular_members_native_cmp_identical=True)])
    digest=hashlib.sha256((json.dumps(receipt,indent=2)+'\n').encode()).hexdigest()
    def mount(value=receipt,pin=digest,limit=1<<20):
        return module.reuse_compressed_view(value,pin,images,work/'view',work,
                                            temporary/'mount',limit,tool,work)
    with pytest.raises(ValueError,match='receipt changed'):
        with mount(pin='0'*64):pytest.fail('changed receipt reached mount')
    with pytest.raises(ValueError,match='storage bound'):
        with mount(limit=1):pytest.fail('storage overflow reached mount')
    image.write_bytes(b'changed image')
    with pytest.raises(ValueError,match='image bytes changed'):
        with mount():pytest.fail('changed image reached mount')
    image.write_bytes(b'original image');archive.write_bytes(b'changed archive')
    with pytest.raises(ValueError,match='image bytes changed'):
        with mount():pytest.fail('changed source reached mount')
    assert not (work/'view').exists() and not (temporary/'mount').exists()


def test_native_output_retirement_obeys_exit_receipt_lock(tmp_path,monkeypatch):
    import fcntl
    import threading
    spec=importlib.util.spec_from_file_location('rrfs_retire_measurement',ROOT/'retire_measurement.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=tmp_path/'run';directory.mkdir()
    names=['output-0.nc','output-1.nc','output-2.nc','output-3.nc']
    for name in names:(directory/name).write_bytes(b'original native words')
    request=dict(runs=['run'],keep=[names[0]],pattern='output-*.nc')
    pending=threading.Event();original=fcntl.flock;results=[]
    with (directory/'native-output-receipt.lock').open('a') as lock:
        original(lock,fcntl.LOCK_EX)
        def observed(handle,operation):
            pending.set()
            return original(handle,operation)
        monkeypatch.setattr(module.fcntl,'flock',observed)
        thread=threading.Thread(target=lambda:results.append(module.retire(tmp_path,request,tmp_path/'retired.json')))
        thread.start()
        assert pending.wait(5)
        (directory/'NATIVE-EXIT.json').write_text('{"exit_status": 0}')
        original(lock,fcntl.LOCK_UN)
    thread.join(5)
    assert not thread.is_alive() and results[0]['completed']
    assert results[0]['retired']=={}
    assert all((directory/name).read_bytes()==b'original native words' for name in names)


def test_streaming_hourly_archive_retires_only_authenticated_intermediates(tmp_path,monkeypatch):
    import hashlib
    import shutil
    import subprocess
    def load(name):
        spec=importlib.util.spec_from_file_location('rrfs_stream_'+name,ROOT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    decode=load('decode_hourly');receiver=load('receive_hourly')
    binary=tmp_path/'ungrib';binary.write_bytes(b'pinned executable')
    table=tmp_path/'Vtable';table.write_bytes(b'pinned workflow table')
    def native_fixture(row,directory,ungrib,vtable):
        assert directory.is_absolute() and ungrib.is_absolute() and vtable.is_absolute()
        directory.mkdir();(directory/'GFS:time').write_bytes(b'original decoded bytes')
        (directory/'Vtable').symlink_to(vtable)
        manifest=dict(row=row,ungrib_sha256=decode.sha(ungrib),vtable_sha256=decode.sha(vtable),
                      intermediates={'GFS:time':decode.sha(directory/'GFS:time')})
        (directory/'MANIFEST.json').write_text(json.dumps(manifest))
    corrupt=False
    def transfer(argv,**kwargs):
        if argv[0]=='rsync' and argv[-1].startswith('archive:'):
            target=Path(argv[-1].removeprefix('archive:'));target.mkdir(parents=True)
            for p in Path(argv[-2]).iterdir():shutil.copy2(p.resolve(),target/p.name)
            if corrupt:(target/'GFS:time').write_bytes(b'changed in transit')
        elif argv[0]=='rsync':Path(argv[-1]).write_bytes(b'verified')
        else:
            import shlex
            args=shlex.split(argv[-1]);options={args[i]:Path(args[i+1]) for i in range(1,len(args),2)}
            publication=receiver.receive(options['--source'],options['--destination'],options['--ready'],json.loads(kwargs['input']))
            return subprocess.CompletedProcess(argv,0,stdout=json.dumps(publication))
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(decode,'decode',native_fixture);monkeypatch.setattr(decode.subprocess,'run',transfer)
    def request(archive):
        return dict(source=dict(host='source',directory='/immutable',rsync_argv=['rsync']),
          ungrib_sha256=decode.sha(binary),vtable_sha256=decode.sha(table),raw_source_publication_sha256='source-pin',interval_seconds=3600,
          rows=[dict(name='hour',valid_time='2000-01-01_00:00:00',inputs=[dict(name='raw',bytes=8,sha256=hashlib.sha256(b'verified').hexdigest())])],
          archive_sink=dict(host='archive',directory=str(archive),rsync_argv=['rsync'],ssh_argv=['ssh'],receiver_argv=['receive'],ready=str(tmp_path/'READY.md')))
    monkeypatch.chdir(tmp_path)
    result=decode.execute(request(tmp_path/'archive'),Path('work'),Path('ungrib'),Path('Vtable'))
    assert result[0]['archive_publication']['files']['GFS:time']==decode.sha(tmp_path/'archive/hour/GFS:time')
    assert not (tmp_path/'work/hour/GFS:time').exists() and not (tmp_path/'work/raw/raw').exists()
    index=tmp_path/'work/INDEX.json'
    content=json.loads(index.read_text())
    content['request']['rows'][0]['inputs'][0]['sha256']='changed source pin'
    index.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='native input contract differs'):
        receiver.finalize(index,tmp_path/'archive',tmp_path/'READY.md')
    assert not (tmp_path/'archive/PUBLICATION.json').exists()
    content['request']=request(tmp_path/'archive');index.write_text(json.dumps(content))
    publication=receiver.finalize(index,tmp_path/'archive',tmp_path/'READY.md')
    assert publication['endpoints']==1
    assert publication['publication_sha256']==decode.sha(tmp_path/'archive/PUBLICATION.json')
    assert not (tmp_path/'archive').stat().st_mode & 0o222
    corrupt=True
    with pytest.raises(ValueError,match='received hourly intermediate bytes changed'):
        decode.execute(request(tmp_path/'bad-archive'),tmp_path/'bad-work',binary,table)
    assert (tmp_path/'bad-work/hour/GFS:time').read_bytes()==b'original decoded bytes'
    assert (tmp_path/'bad-work/raw/raw').exists() and not (tmp_path/'bad-archive/hour').exists()


def test_additional_geography_authenticates_all_members_before_linking(tmp_path):
    import importlib.util
    import json
    import pytest
    spec = importlib.util.spec_from_file_location('rrfs_dataset_dependencies', ROOT/'stage_geography.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    view = tmp_path/'view'
    view.mkdir()
    manifest = view/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(completed=True, original=True))+'\n')
    source = tmp_path/'original'
    source.mkdir()
    (source/'index').write_bytes(b'original native dataset metadata')
    (source/'tile').write_bytes(b'original native geography bytes')
    request = dict(source=str(source), files={name:dict(sha256=module.sha(source/name),
        bytes=(source/name).stat().st_size) for name in ['index','tile']},
        authority='original source inventory', source_receipt_sha256='a'*64)
    original_sha = module.sha(manifest)
    (source/'tile').write_bytes(b'changed native geography bytes')
    with pytest.raises(ValueError, match='supplemental geography bytes changed'):
        module.add_pinned_datasets(view, {'additional':request})
    assert not (view/'additional').exists() and module.sha(manifest)==original_sha
    (source/'tile').write_bytes(b'original native geography bytes')
    result = module.add_pinned_datasets(view, {'additional':request})
    assert (view/'additional').is_symlink()
    assert (view/'additional/tile').read_bytes()==(source/'tile').read_bytes()
    assert result['base_view_manifest_sha256']==original_sha
    assert result['supplemental_datasets']['additional']['files']==request['files']
    with pytest.raises(ValueError, match='cannot replace an existing dataset'):
        module.add_pinned_datasets(view, {'additional':request})


@pytest.mark.parametrize('defect', [None, 'lost-stage', 'omitted-endpoint', 'clock-gap',
    'wrong-static', 'changed-companion', 'changed-publication', 'chemistry-on', 'lost-ruling'])
def test_canonical_catalogue_binds_complete_cases_and_companions(tmp_path,monkeypatch,defect):
    monkeypatch.syspath_prepend(str(ROOT))
    spec=importlib.util.spec_from_file_location('rrfs_canonical_catalogue',ROOT/'canonical_catalogue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from ruled_reference import authority, INIT_BINARY
    approved=authority('init',INIT_BINARY)
    def pinned_file(name):
        p=tmp_path/name;p.write_bytes(name.encode());return dict(path=str(p),sha256=module.sha(p))
    companions={n:pinned_file(n) for n in ['grid','static.nc','invariant.nc','ugwp_oro_data.nc']}
    product=tmp_path/'product';product.mkdir()
    rows=[];witness=[]
    for num,case in enumerate([7,9,9]):
        directory=tmp_path/f'case-{num}';directory.mkdir()
        output=directory/'native.nc';output.write_bytes(f'fixture weather bytes {num}'.encode())
        config=directory/'namelist.init_atmosphere';config.write_text("&chemistry config_smoke_scheme='off' /\n")
        time='2000-01-01_'+('01' if num==2 else '00')+':00:00'
        manifest=dict(row=dict(variables=dict(init_case=case,start_time=time,nvertlevels=59,nsoillevels=9,decomp_file_prefix='fixture.part.')),
            completed=True,exit_status=0,binary_sha256=INIT_BINARY,source_correction=approved,reference_label=approved['label'],
            source_pins=dict(source_correction=approved),inputs={n:companions[n]['sha256'] for n in ['static.nc','invariant.nc']},
            configuration={config.name:module.sha(config)},outputs={output.name:module.sha(output)})
        if defect=='chemistry-on' and num==0:
            config.write_text("&chemistry config_smoke_scheme='on' /\n");manifest['configuration'][config.name]=module.sha(config)
        if defect=='lost-ruling' and num==0:manifest['source_pins']['source_correction']={}
        (directory/'MANIFEST.json').write_text(json.dumps(manifest))
        members={n:module.sha(directory/n) for n in ['MANIFEST.json',config.name,output.name]}
        (directory/'PUBLICATION.json').write_text(json.dumps(dict(files=members)))
        row=dict(path=str(directory),publication_sha256=module.sha(directory/'PUBLICATION.json'),outputs=manifest['outputs'],floating_words=1,nonfinite_words=0)
        rows.append(row);witness.append(dict(**row,authority=approved))
    (product/'INDEX.json').write_text(json.dumps(dict(complete=True,nonfinite_words=0,source_correction=approved,stages=3,products=rows,floating_words=3,netcdf_files=3)))
    (product/'VERIFIED.json').write_text(json.dumps(dict(verified=True,forecast_identity=False,cases=witness)))
    (product/'PUBLICATION.json').write_text(json.dumps(dict(files={n:module.sha(product/n) for n in ['INDEX.json','VERIFIED.json']})))
    request=dict(input_products=[dict(path=str(product),publication_sha256=module.sha(product/'PUBLICATION.json'))],
        case_groups=[dict(id='fixture',mesh='fixture',decomp_prefix='fixture.part.',case7=rows[0]['path'],case9=[r['path'] for r in rows[1:]],
                          companions=companions,boundary_lead_seconds=[0,3600],boundary_interval_seconds=3600)],
        forecast_tables={'table':pinned_file('table')},deck_indexes=[pinned_file('deck-index')])
    group=request['case_groups'][0]
    if defect=='lost-stage':group['case9']=[];group['boundary_lead_seconds']=[]
    elif defect=='omitted-endpoint':group['case9']=group['case9'][:1];group['boundary_lead_seconds']=[0]
    elif defect=='clock-gap':group['boundary_lead_seconds']=[0,7200]
    elif defect=='wrong-static':group['companions']['static.nc']=pinned_file('unrelated-static')
    elif defect=='changed-companion':Path(companions['grid']['path']).write_bytes(b'changed')
    elif defect=='changed-publication':(Path(rows[0]['path'])/'PUBLICATION.json').write_text('{}')
    if defect:
        with pytest.raises(ValueError):module.assemble(request)
    else:
        result=module.assemble(request)
        assert result['stages']==3 and result['groups']==1 and result['netcdf_files']==3
        assert result['source_correction']==approved
        assert result['selected_companions_rehashed'] and not result['native_weather_outputs_rehashed']
        assert not result['weather_execution'] and not result['module_replay'] and not result['forecast_identity']
