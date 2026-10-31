#!/usr/bin/env python3
"""Extend f0 deck generation with fork registries and the workflow templates.

All runtime choices come from explicit JSON, with no mesh/date defaults.
Included TEMPO, Noah-MP, diagnostics and smoke registries are source authority.
The native executable remains responsible for option validation and activation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
from string import Template
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mpas_f0_decks', HERE.parent / 'generate_decks.py')
f0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f0)


def registry(source):
    root = source / 'src/core_atmosphere'
    files = set()

    def expand(path):
        files.add(path)
        text = re.sub(r'<!--.*?-->', '', path.read_text(), flags=re.S)
        def include(match):
            relative = path.parent / match[1]
            return expand(relative if relative.is_file() else root / match[1])
        text = re.sub(r'^\s*#include\s+"([^"]+)".*$', include, text, flags=re.M)
        return re.sub(r'^\s*#.*$', '', text, flags=re.M)

    tree = ET.fromstring(expand(root / 'Registry.xml'))
    groups = {}
    for record in tree.iter('nml_record'):
        for option in record.findall('nml_option'):
            groups[option.get('name')] = (record.get('name'), option.attrib)
    return groups, files


def read_namelist(text):
    groups = {}
    group = None
    for line in text.splitlines():
        line = line.split('!', 1)[0].strip()
        if not line:
            continue
        if line.startswith('&'):
            group = line[1:]
            groups.setdefault(group, {})
        elif line == '/':
            group = None
        else:
            key, value = line.split('=', 1)
            if group is None:
                raise ValueError('assignment outside namelist group')
            groups[group][key.strip()] = value.strip().rstrip(',')
    return groups


def literal(value, kind):
    if kind == 'logical':
        if not isinstance(value, bool):
            raise ValueError('logical override requires JSON boolean')
        return '.true.' if value else '.false.'
    if kind == 'integer':
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError('integer override required')
        return str(value)
    if kind == 'real':
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError('real override required')
        return str(value)
    return f0.literal(value)


def decoded(value, kind):
    if kind == 'character':
        return value.strip("'\"")
    if kind == 'logical':
        return value.strip('.').lower() == 'true'
    return int(value) if kind == 'integer' else float(value)


def companions(overrides, rules):
    result = {}
    for rule in rules:
        if any(overrides.get(key) != expected for key, expected in rule.get('when', {}).items()):
            continue
        value = overrides.get(rule['option'])
        if value is not None and (value == rule.get('value') or
                                 ('except' in rule and value != rule['except'])):
            result.update(rule['settings'])
    # Explicit selections take precedence; the table only fills companions.
    return {key: value for key, value in result.items() if key not in overrides}


def generate(source, workflow, out, runtime, specifications):
    f0.validate_cadence(runtime['duration'], runtime['output_interval'], runtime['dt'])
    diagnostic_interval = runtime.get('diagnostic_interval', runtime['output_interval'])
    f0.validate_cadence(runtime['duration'], diagnostic_interval, runtime['dt'])
    groups, registry_files = registry(source)
    choices, suites = f0.options(source)
    for name, (_, attr) in groups.items():
        if name.endswith('_scheme') and name not in choices:
            values = re.findall(r"[`']([a-z][a-z0-9_]*)[`']", attr.get('possible_values', ''))
            if values:
                choices[name] = sorted(set(values))
    substitutions = dict(runtime['workflow_variables'])
    substitutions.update(dt=runtime['dt'], substeps=runtime['substeps'], start_time=runtime['start'],
                         run_duration=runtime['duration'], MESH_NAME=runtime['mesh'])
    template = workflow / 'parm/hrrrv5/namelist.atmosphere'
    base = read_namelist(Template(template.read_text()).substitute(substitutions))
    opaque = {name: values for name, values in base.items() if all(key not in groups for key in values)}
    workflow_settings = {key: decoded(value, groups[key][1]['type'])
                         for name, record in base.items() if name not in opaque for key, value in record.items()}
    specs = [dict(name='suite-' + suite, suite=suite, override={}) for suite in suites]
    specs += [dict(name=name.removeprefix('config_') + '-' + value.strip('.'), suite='none',
                   override={name: decoded(value, 'logical') if value.startswith('.') else value})
              for name, values in choices.items() for value in values]
    for variant in specifications['workflow_option_variants']:
        for key, value in workflow_settings.items():
            if key.startswith(variant['prefix']):
                specs.append(dict(name=variant['value'] + '-' + key.removeprefix('config_'), suite='none',
                                  override={variant['scheme']: variant['value'], key: value}))
    specs += specifications['variants']
    if len({row['name'] for row in specs}) != len(specs):
        raise ValueError('duplicate deck specification')
    if specifications.get('selected_names') is not None:
        selected = set(specifications['selected_names'])
        if not selected or not selected.issubset({row['name'] for row in specs}):
            raise ValueError('deck selection must name existing source-derived rows')
        specs = [row for row in specs if row['name'] in selected]
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    stream_template = workflow / 'parm/streams.atmosphere'
    for row in specs:
        override = row['override']
        companion = companions(override, specifications['companion_rules'])
        settings = dict(workflow_settings)
        settings.update(suites[row['suite']])
        settings.update(config_physics_suite=row['suite'])
        settings.update(companion)
        settings.update(override)
        settings['num_soil_layers'] = specifications['soil_levels'][settings['config_lsm_scheme']]
        settings['config_apply_lbcs'] = runtime['limited_area']
        settings['config_block_decomp_file_prefix'] = runtime['graph_prefix']
        if 'do_restart' in runtime:
            settings['config_do_restart'] = runtime['do_restart']
        records = {key: dict(opaque.get(key, {})) for key in base}
        for key, value in settings.items():
            group, attr = groups[key]
            records.setdefault(group, {})[key] = literal(value, attr['type'])
        deck = out / row['name']
        deck.mkdir()
        nml = deck / 'namelist.atmosphere'
        nml.write_text(''.join('&' + name + '\n' + ''.join(' ' + k + ' = ' + v + '\n' for k, v in values.items())
                               + '/\n' for name, values in records.items()))
        stream_values = dict(mpasout_interval=runtime['output_interval'], restart_interval=runtime['restart_interval'],
                             history_interval=runtime['output_interval'], diag_interval=diagnostic_interval,
                             lbc_interval=runtime['lbc_interval_hours'])
        text = stream_template.read_text()
        for key, value in stream_values.items():
            text = text.replace('@' + key + '@', str(value))
        if re.search(r'@\w+@', text):
            raise ValueError('unresolved stream placeholder')
        streams = ET.fromstring(text)
        for stream in streams:
            if stream.get('name') in runtime['input_filenames']:
                stream.set('filename_template', runtime['input_filenames'][stream.get('name')])
            for listing in stream.findall('file'):
                path = Path(listing.get('name'))
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('unsafe stream list filename')
                target = deck / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(workflow / 'fix/stream_list/hrrrv5' / path.name, target)
        streamfile = deck / 'streams.atmosphere'
        ET.ElementTree(streams).write(streamfile, encoding='unicode')
        entries.append(dict(deck=row['name'], suite=row['suite'], override=override, companions=companion,
                            resolved_settings=settings,
                            files={str(p.relative_to(deck)): f0.sha(p) for p in sorted(deck.rglob('*')) if p.is_file()}))
    sources = registry_files | {source / f0.CONTROL, template, stream_template}
    index = dict(schema='mpas.rrfs-native-decks/v1', runtime=runtime,
                 source_sha256={str(p): f0.sha(p) for p in sorted(sources)},
                 specifications=specifications, workflow_unregistered_groups=opaque,
                 option_choices=choices, decks=entries)
    (out / 'INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'workflow', 'out', 'runtime'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--specifications', type=Path, default=HERE / 'deck-specs.json')
    a = p.parse_args()
    index = generate(a.source, a.workflow, a.out, json.loads(a.runtime.read_text()),
                     json.loads(a.specifications.read_text()))
    print(json.dumps({'decks': len(index['decks']), 'index': str(a.out / 'INDEX.json')}))
