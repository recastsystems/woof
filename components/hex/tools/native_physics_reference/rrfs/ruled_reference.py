#!/usr/bin/env python3
"""Build/seal exactly the owner's two approved, separately labelled authorities.

The ruling record itself carries private context, so it is not in this tree: set
RRFS_SOURCE_AUTHORITY_RULING to the reference store's copy (beside the init derivative build,
native/derivatives/<derivative>/DREW-RULING.json). Its bytes stay bound by SHA-256 below.
"""
import argparse
import hashlib
import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from publish_cache import publish, sha
from rrfs.prepare_build import prepare, PINS
from rrfs.publish_diagnostic_build import verify_sources
from verify_native_reference import verify_commands
from verify_diagnostic_contract import publication

RULING = Path(os.environ.get('RRFS_SOURCE_AUTHORITY_RULING')
              or HERE/'diagnostics/source-authority-ruling.json')
PATCHES = {'init': 'init-optional-aerosol-nullify.patch',
           'gfl': 'native-gfl-state-authority-proposal.patch'}
SOURCE_PATHS = {'init': 'src/core_init_atmosphere/mpas_init_atm_cases.F',
               'gfl': 'src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_mpas.F90'}
PATCH_HASHES = {'init': '424133d89247f88603c6da5dd37a94755a6b81c43b85bbc0bffc5348ed84750c',
                'gfl': '8dcf2f4c31501f3a72a72f72f250383290bfa95df1130179c2986698ddaeb157'}
INIT_BINARY = '8d0e8c69f39ab194a6574cba51ac9135499e4eed977ca193fa19cbca51481845'


def authority(kind, binary_sha256):
    if not RULING.is_file():
        raise ValueError('the 2026-10-14 07:45Z source-authority ruling record is not in this checkout; '
                         'set RRFS_SOURCE_AUTHORITY_RULING to the reference store copy')
    if sha(RULING) != 'b5b90b2290c44a1870a120330515a42c54b44b7ae4b986a9f3df0b4eb2a39fae':
        raise ValueError('recorded owner ruling bytes changed')
    ruling = json.loads(RULING.read_text())
    if kind not in PATCHES or ruling['fork'] != PINS['fork'] or (
            ruling['verbatim'] != 'yes to both rrfs stuff'):
        raise ValueError('unsupported approved source authority')
    patch = HERE/'diagnostics'/PATCHES[kind]
    if sha(patch) != PATCH_HASHES[kind] or ruling[kind]['patch_sha256'] != PATCH_HASHES[kind]:
        raise ValueError('approved patch bytes changed')
    if kind == 'init' and binary_sha256 != INIT_BINARY:
        raise ValueError('approved initializer binary differs')
    return dict(kind=kind, label=ruling[kind]['label'], patch_sha256=sha(patch),
                binary_sha256=binary_sha256, ruling_sha256=sha(RULING), ruling=ruling)


def validate_authority(value, binary_sha256=None):
    expected = authority(value['kind'], value['binary_sha256'])
    if value != expected or (binary_sha256 and value['binary_sha256'] != binary_sha256):
        raise ValueError('source correction/ruling/binary binding differs')
    return value


def validate_init_case(receipt):
    value = receipt.get('source_correction') or receipt.get('source_pins', {}).get('source_correction')
    if value:
        validate_authority(value, receipt['binary_sha256'])
        if value['kind'] != 'init' or receipt['row']['variables']['init_case'] not in (7,9):
            raise ValueError('approved initializer is limited to canonical case 7/9')
        if receipt.get('source_correction') != value or receipt['source_pins'].get('source_correction') != value:
            raise ValueError('case source/ruling binding differs')
    return value


def validate_init_configuration(text):
    """The approved initializer derivative generates canonical aerosol-off inputs."""
    # Remove comments while preserving exclamation marks inside quoted strings.
    text = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|!.*$",
                  lambda m:'' if m.group().startswith('!') else m.group(), text, flags=re.M)
    for option in ('config_smoke_scheme', 'config_dust_scheme'):
        values = re.findall(r'\b' + option + r"\s*=\s*['\"]([^'\"]+)['\"]", text, re.I)
        # 1d6e370e src/core_init_atmosphere/registry.chemistry.xml:24-29
        # declares both defaults as off, in_defaults=false. The pristine
        # workflow therefore intentionally omits these namelist entries.
        if len(values) > 1 or (values and values[0] != 'off'):
            raise ValueError('approved initializer requires canonical aerosol-off configuration: ' + option)


def validate_init_authorities(values):
    """Validate inherited initializer rulings without widening their authority."""
    if not isinstance(values, list):
        raise ValueError('inherited initializer authorities must be a list')
    for value in values:
        validate_authority(value)
        if value['kind'] != 'init':
            raise ValueError('inherited input authority is not an initializer derivative')
    return values


def input_authorities(case):
    """Recover initializer authority from the actual selected input manifests."""
    found = {}
    for source in case['files'].values():
        path = Path(source)
        metadata = path.parent/'MANIFEST.json'
        if metadata.is_file():
            receipt = json.loads(metadata.read_text())
            if receipt.get('schema') == 'mpas.rrfs-native-input/v1':
                value = validate_init_case(receipt)
                if receipt.get('source_pins',{}).get('diagnostic'):
                    raise ValueError('diagnostic initializer input lacks approved case authority')
                if value:
                    if receipt['outputs'].get(path.name) != sha(path):
                        raise ValueError('selected derivative input differs from approved case')
                    found[value['binary_sha256']] = value
            elif receipt.get('schema') == 'mpas.rrfs-native-forecast/v1':
                values = validate_init_authorities(receipt.get('init_derivative_authorities', []))
                if authority('init', INIT_BINARY)['label'] in receipt.get('reference_labels', []) and not values:
                    raise ValueError('restart forecast initializer binding missing')
                if values:
                    if not receipt.get('completed') or receipt.get('exit_status') != 0 or (
                            receipt.get('outputs', {}).get(path.name) != sha(path)):
                        raise ValueError('selected restart differs from completed derivative forecast')
                    if any(v['label'] not in receipt.get('reference_labels', []) for v in values):
                        raise ValueError('restart forecast derivative label missing')
                    found.update({v['binary_sha256']: v for v in values})
        sidecar = path.parent/'INPUT-AUTHORITIES.json'
        if sidecar.is_file():
            transported = json.loads(sidecar.read_text())
            if transported.get('schema') != 'mpas.rrfs-transported-input-authority/v1':
                raise ValueError('transported input authority schema differs')
            row = transported['files'].get(path.name)
            if row is None or row['sha256'] != sha(path):
                raise ValueError('transported input authority bytes missing or changed')
            values = validate_init_authorities(row['init_derivative_authorities'])
            found.update({v['binary_sha256']: v for v in values})
    result = [found[k] for k in sorted(found)]
    if case.get('init_derivative_authorities',result) != result:
        raise ValueError('declared derivative input authority differs from actual files')
    return result


def corrections(source, kind):
    relative = SOURCE_PATHS[kind]
    module = 'src/core_atmosphere/physics/physics_noaa/GFL' if kind == 'gfl' else ''
    name = str(Path(relative).relative_to(module)) if module else relative
    commit = PINS['submodules'][module] if module else PINS['fork']
    original = subprocess.check_output(['git', '-C', str(source/module), 'show', commit+':'+name])
    if kind == 'gfl':
        old = b'       ierr(i)     = 0\n'
        if original.count(old) != 1:
            raise ValueError('GFL initialization location changed')
        corrected = original.replace(old, old+b'       ierrm(i)    = 0\n')
    else:
        corrected = original
        for old in (b"      call mpas_pool_get_array(fg, 'p', p_fg)\n! Chemistry\n",
                    b"      call mpas_pool_get_array(fg, 'qnr', qnr_fg)\n! Chemistry\n"):
            if corrected.count(old) != 1:
                raise ValueError('initializer NULLIFY location changed')
            corrected = corrected.replace(old, old.replace(b'! Chemistry\n',
                b'      nullify(smoke_fine_fg, dust_fine_fg, dust_coarse_fg)\n! Chemistry\n'))
    return {relative:dict(original_sha256=hashlib.sha256(original).hexdigest(),
                         corrected_sha256=hashlib.sha256(corrected).hexdigest())}


def prepare_gfl(source, destination, tables):
    authority('gfl', 'pending-build')
    prepare(source, destination, 'gnu-strict', tables)
    patch = HERE/'diagnostics'/PATCHES['gfl']
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=destination, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=destination, check=True)
    expected = corrections(source, 'gfl')
    if any(sha(destination/name) != row['corrected_sha256'] for name,row in expected.items()):
        raise ValueError('applied source correction differs')
    pins_path = destination/'SOURCE-PINS.json'
    pins = json.loads(pins_path.read_text())
    pins['source_correction'] = authority('gfl', 'pending-build')
    pins_path.write_text(json.dumps(pins, indent=2)+'\n')
    with (destination/'build-source.diff').open('a') as stream:
        stream.write(patch.read_text())


def stage_build(build, source, stage, kind):
    core = 'init_atmosphere' if kind == 'init' else 'atmosphere'
    binary = build/(core+'_model')
    approved = authority(kind, sha(binary))
    pins = json.loads((build/'SOURCE-PINS.json').read_text())
    for key in ('fork','workflow','submodules','mpas_data','precision','musica'):
        if pins.get(key) != PINS[key]:
            raise ValueError('corrected build source pins differ: '+key)
    allowed_legacy = 'counterfactual-optional-aerosol-nullify' if kind == 'init' else None
    if pins.get('diagnostic') != allowed_legacy or pins['recipe'] != 'gnu-strict':
        raise ValueError('unapproved compiler diagnostic or recipe')
    audit = json.loads((build/'flag-audit.json').read_text())
    records = [json.loads(line) for line in (build/'compile-commands.jsonl').read_text().splitlines()]
    manifest = dict(schema='mpas.rrfs-native-build/v1', core=core, recipe='gnu-strict',
        reference='init-derivative' if kind=='init' else 'source-corrected',
        pin_set={key:PINS[key] for key in ('fork','workflow','submodules','mpas_data','precision','musica')},
        source_correction=approved, binary_sha256=sha(binary),
        toolchain_sha256=sha(build/'toolchain.txt'), flags_sha256=sha(build/'flag-audit.json'),
        source_diff_sha256=sha(build/'build-source.diff'),
        objects=len(audit['objects']), groups=audit['groups'], forecast_identity='unmeasured')
    manifest['pin_set_hash'] = hashlib.sha256(json.dumps(manifest['pin_set'],sort_keys=True).encode()).hexdigest()
    verification = verify_commands(audit, manifest, records)
    actual_objects = {str(p.relative_to(build)):sha(p) for p in build.rglob('*.o')}
    if actual_objects != {r['object']:r['sha256'] for r in audit['objects']}:
        raise ValueError('current object inventory or bytes differ from captured audit')
    verification['raw_objects_rehashed'] = True
    source_check = verify_sources(source, build, pins, corrections(source, kind))
    patch = HERE/'diagnostics'/PATCHES[kind]
    if not (build/'build-source.diff').read_bytes().endswith(patch.read_bytes()):
        raise ValueError('source diff does not bind exact approved patch')
    stage.mkdir(parents=True, exist_ok=False)
    names = [binary.name,'flag-audit.json','compile-commands.jsonl','toolchain.txt',
             'build-source.diff','build.log']
    if core == 'atmosphere':
        names += ['build_tables','build_tables_tempo_v3']
        manifest['utility_sha256'] = {n:sha(build/n) for n in names[-2:]}
    for name in names:
        shutil.copy2(build/name,stage/name)
    shutil.copy2(build/'SOURCE-PINS.json',stage/'ORIGINAL-SOURCE-PINS.json')
    pins.update(diagnostic=None, source_correction=approved)
    (stage/'SOURCE-PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
    for name,value in [('MANIFEST.json',manifest),('SOURCE-VERIFICATION.json',source_check),
                       ('OBJECT-VERIFICATION.json',verification)]:
        (stage/name).write_text(json.dumps(value,indent=2)+'\n')
    shutil.copy2(patch,stage/'APPROVED.patch')
    shutil.copy2(RULING,stage/'DREW-RULING.json')
    return manifest


def publish_stage(stage, reference):
    manifest = json.loads((stage/'MANIFEST.json').read_text())
    approved = validate_authority(manifest['source_correction'],sha(stage/(manifest['core']+'_model')))
    if sha(stage/'DREW-RULING.json') != approved['ruling_sha256'] or sha(stage/'APPROVED.patch') != approved['patch_sha256']:
        raise ValueError('transported ruling or approved patch differs')
    audit = json.loads((stage/'flag-audit.json').read_text())
    records = [json.loads(line) for line in (stage/'compile-commands.jsonl').read_text().splitlines()]
    verify_commands(audit,manifest,records)
    pins = json.loads((stage/'SOURCE-PINS.json').read_text())
    if pins.get('diagnostic') or pins.get('source_correction') != approved:
        raise ValueError('corrected publication source authority differs')
    source_check = json.loads((stage/'SOURCE-VERIFICATION.json').read_text())
    if not source_check['all_other_pinned_code_bytes_identical'] or (
            set(source_check['corrections']) != {SOURCE_PATHS[approved['kind']]}):
        raise ValueError('source correction scope differs')
    bindings = {'toolchain.txt':'toolchain_sha256','flag-audit.json':'flags_sha256',
                'build-source.diff':'source_diff_sha256'}
    bindings.update({n:None for n in manifest.get('utility_sha256',{})})
    for name,key in bindings.items():
        if sha(stage/name) != (manifest[key] if key else manifest['utility_sha256'][name]):
            raise ValueError('corrected build member differs: '+name)
    key = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return publish(stage,reference/'native/derivatives'/key,
        [p.name for p in stage.iterdir() if p.is_file() and p.name!='PUBLICATION.json'],
        reference/'READY.md',approved['label']+'; strict single, exact approved patch, '+
        str(manifest['objects'])+' audited objects and complete source verification; '+
        'pristine authorities retained; run using recorded MPICH/toolchain; forecast identity unmeasured')


def authorize_existing_case(source, build, destination, ready, evidence):
    """Authorize archived, unchanged case bytes; retain the historical receipt."""
    from publish_cache import seal
    original = publication(source/'PUBLICATION.json',sha(source/'PUBLICATION.json'))
    publication(build/'PUBLICATION.json',sha(build/'PUBLICATION.json'))
    native = json.loads((build/'MANIFEST.json').read_text())
    approved = validate_authority(native['source_correction'],native['binary_sha256'])
    if approved['kind'] != 'init':
        raise ValueError('case authorization requires approved init derivative')
    old = json.loads((source/'MANIFEST.json').read_text())
    if not old['completed'] or old['exit_status'] or old['binary_sha256'] != approved['binary_sha256'] or (
            old['audit_sha256'] != native['flags_sha256']):
        raise ValueError('historical case producer differs from approved init authority')
    if old['source_pins'].get('diagnostic') != 'counterfactual-optional-aerosol-nullify' or (
            old['source_pins'].get('diagnostic_source_patch_sha256') != approved['patch_sha256']):
        raise ValueError('historical case patch differs from approved correction')
    for key,value in native['pin_set'].items():
        if old['source_pins'].get(key) != value:
            raise ValueError('historical case source pins differ')
    updated = dict(old, historical_authorization=('Existing unchanged bytes approved by '
                       + approved['ruling']['authority'].split(';')[0] + ' ruling '
                       + approved['ruling']['date'] + '; no new execution'),
                   original_publication_sha256=sha(source/'PUBLICATION.json'),
                   original_manifest_sha256=sha(source/'MANIFEST.json'),
                   source_correction=approved,reference_label=approved['label'],
                   source_pins=json.loads((build/'SOURCE-PINS.json').read_text()),
                   init_build_publication=dict(path=str(build),publication_sha256=sha(build/'PUBLICATION.json')))
    validate_init_case(updated)
    for name,h in old['outputs'].items():
        if original['files'].get(name) != h or sha(source/name) != h:
            raise ValueError('archived native case output differs')
    if any(sha(source/name) != h for name,h in old['configuration'].items()):
        raise ValueError('archived native case configuration differs')
    for path,digest in evidence.items():
        if sha(Path(path)) != digest:
            raise ValueError('original case comparison or finiteness evidence differs')
    temporary = destination.with_name(destination.name+'.partial')
    temporary.mkdir(parents=True,exist_ok=False)
    for name in old['outputs']:
        if (source/name).stat().st_dev != temporary.stat().st_dev:
            raise ValueError('canonical case aliases must share archive storage device')
        os.link(source/name,temporary/name)
    for name in old['configuration']:
        shutil.copy2(source/name,temporary/name)
    shutil.copy2(source/'MANIFEST.json',temporary/'ORIGINAL-MANIFEST.json')
    shutil.copy2(source/'PUBLICATION.json',temporary/'ORIGINAL-PUBLICATION.json')
    shutil.copy2(RULING,temporary/'DREW-RULING.json')
    shutil.copy2(HERE/'diagnostics'/PATCHES['init'],temporary/'APPROVED.patch')
    for path in evidence:
        shutil.copy2(path,temporary/Path(path).name)
    (temporary/'MANIFEST.json').write_text(json.dumps(updated,indent=2)+'\n')
    result=dict(schema='mpas.native-cache-publication/v1',description=approved['label']+
                '; canonical case '+str(old['row']['variables']['init_case'])+
                '; archived unchanged outputs with historical evidence; not pristine; no new forecast',
                source=str(source),selection=['explicit authenticated outputs and metadata'],
                files={p.name:sha(p) for p in temporary.iterdir() if p.is_file()})
    (temporary/'PUBLICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    temporary.rename(destination)
    seal(destination)
    digest=sha(destination/'PUBLICATION.json')
    with ready.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(result['description']+': `'+str(destination)+'`; PUBLICATION.json SHA256 `'+digest+'`.\n')
    return dict(path=str(destination),publication_sha256=digest,files=len(result['files']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare-gfl','stage','publish'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--tables',type=Path)
    parser.add_argument('--kind',choices=tuple(PATCHES))
    parser.add_argument('--reference',type=Path)
    a = parser.parse_args()
    if a.action=='prepare-gfl': prepare_gfl(a.source,a.stage,a.tables)
    elif a.action=='stage': print(json.dumps(stage_build(a.build,a.source,a.stage,a.kind)))
    else: print(json.dumps(publish_stage(a.stage,a.reference)))
