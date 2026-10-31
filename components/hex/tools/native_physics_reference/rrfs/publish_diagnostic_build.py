#!/usr/bin/env python3
"""Seal a source-unchanged checked native build as diagnostic evidence only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha
from rrfs.diagnostic_flags import DIAGNOSTICS, INTEGER_POISON, checked_flags

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / 'pins.json').read_text())
spec = importlib.util.spec_from_file_location('rrfs_diagnostic_audit', HERE / 'audit_flags.py')
flags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flags)
CODE_SUFFIXES = {'.f', '.f90', '.f95', '.f03', '.f08', '.c', '.cc', '.cpp', '.h', '.inc'}


def verify_sources(source, build, pins, corrections=None):
    """Compare original code bytes directly with each pinned Git archive."""
    repositories = [('', pins['fork']), *pins['submodules'].items()]
    sources, groups = {}, {}
    for relative, commit in repositories:
        repository, destination = source / relative, build / relative
        actual = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain',
                                        '--untracked-files=no'], text=True)
        if actual != commit or dirty:
            raise ValueError('diagnostic source repository differs from the pin: ' + relative)
        process = subprocess.Popen(['git', '-C', str(repository), 'archive', commit], stdout=subprocess.PIPE)
        count = 0
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    if not member.isfile() or Path(member.name).suffix.lower() not in CODE_SUFFIXES:
                        continue
                    stream = archive.extractfile(member)
                    expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                    target = destination / member.name
                    correction = (corrections or {}).get(str(target.relative_to(build)))
                    if correction:
                        if correction['original_sha256'] != expected:
                            raise ValueError('approved correction original source differs')
                        expected = correction['corrected_sha256']
                    if not target.is_file() or sha(target) != expected:
                        raise ValueError('native diagnostic source bytes changed: ' + str(target))
                    sources[str(target.relative_to(build))] = expected
                    count += 1
        except BaseException:
            process.terminate()
            raise
        finally:
            process.stdout.close()
            status = process.wait()
        if status or not count:
            raise ValueError('pinned code archive incomplete: ' + relative)
        groups[relative or 'fork'] = count
    if corrections and not set(corrections).issubset(sources):
        raise ValueError('approved correction absent from compiled source inventory')
    result = dict(schema='mpas.rrfs-pristine-diagnostic-source/v1', sources=sources,
                  groups=groups, all_pinned_code_bytes_identical=not bool(corrections))
    if corrections:
        result.update(schema='mpas.rrfs-approved-corrected-source/v1',
                      corrections=corrections, all_other_pinned_code_bytes_identical=True)
    return result


def publish_diagnostic(build, source, reference, core):
    pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    diagnostic = pins.get('diagnostic')
    if diagnostic not in DIAGNOSTICS or pins.get('recipe') != 'gnu-strict' or (
            pins.get('reference', 'primary') != 'primary'):
        raise ValueError('only named primary strict checked diagnostic builds are supported')
    for key in ('fork', 'workflow', 'submodules', 'mpas_data', 'precision', 'musica'):
        if pins.get(key) != PINS[key]:
            raise ValueError('diagnostic pin set changed: ' + key)
    audited = flags.audit(build.resolve(strict=True), 'gnu-strict', core)
    saved = json.loads((build / 'flag-audit.json').read_text())
    if not audited['all_ok'] or saved != audited:
        raise ValueError('every checked Fortran object and original audit must agree')
    provenance = verify_sources(source, build, pins)
    (build / 'PRISTINE-SOURCES.json').write_text(json.dumps(provenance, indent=2) + '\n')
    binary = build / (core + '_model')
    manifest = dict(schema='mpas.rrfs-checked-native-diagnostic-build/v1',
        diagnostic=diagnostic, reference_binary=False, native_source_unchanged=True,
        core=core, recipe='gnu-strict', precision='single', musica=False,
        pin_set={key: pins[key] for key in ('fork', 'workflow', 'submodules', 'mpas_data', 'precision', 'musica')},
        binary_sha256=sha(binary), audit_sha256=sha(build / 'flag-audit.json'),
        toolchain_sha256=sha(build / 'toolchain.txt'),
        pristine_sources_sha256=sha(build / 'PRISTINE-SOURCES.json'),
        source_diff_sha256=sha(build / 'build-source.diff'),
        objects=len(audited['objects']), groups=audited['groups'],
        recipe_tools_sha256={name: sha(HERE / name) for name in
            ('prepare_build.py', 'audit_flags.py', 'build_reference.sh', 'publish_diagnostic_build.py',
             'diagnostic_flags.py')},
        checked_fortran_flags=list(checked_flags(diagnostic)),
        affected_word_set_proven=False, exclusions_authorized=False)
    if diagnostic == 'bounds-snan-integer-poison':
        manifest['integer_poison'] = INTEGER_POISON
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (build / 'DIAGNOSTIC-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return publish(build, reference / 'diagnostics' / key,
        [binary.name, 'DIAGNOSTIC-MANIFEST.json', 'PRISTINE-SOURCES.json', 'SOURCE-PINS.json',
         'flag-audit.json', 'compile-commands.jsonl', 'toolchain.txt', 'build-source.diff',
         'build.log', 'executable.sha256'], reference / 'READY.md',
        'DIAGNOSTIC ONLY: primary fork source-unchanged strict single ' + core +
        ' with every Fortran object ' + ' '.join(checked_flags(diagnostic)) +
        '; compare against pristine native on the same host/rank. '
        'Not a reference executable, affected-word witness, repair, exclusion or forecast acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--core', required=True, choices=('atmosphere', 'init_atmosphere'))
    args = parser.parse_args()
    print(json.dumps(publish_diagnostic(args.build, args.source, args.reference, args.core)))
