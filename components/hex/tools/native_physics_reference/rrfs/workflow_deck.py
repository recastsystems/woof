#!/usr/bin/env python3
"""Render an explicit workflow stream overlay on an authenticated source deck."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location('rrfs_workflow_deck_generator', Path(__file__).with_name('generate_decks.py'))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def generate(source_deck, workflow, source, out, request):
    index_path = source_deck.parent / 'INDEX.json'
    index = json.loads(index_path.read_text())
    row = copy.deepcopy(next(row for row in index['decks'] if row['deck'] == source_deck.name))
    if any(generator.f0.sha(source_deck / name) != digest for name, digest in row['files'].items()):
        raise ValueError('source deck differs from its immutable index')
    groups, _ = generator.registry(source)
    namelist = generator.read_namelist((source_deck / 'namelist.atmosphere').read_text())
    for name, value in request['settings'].items():
        group, attr = groups[name]
        namelist.setdefault(group, {})[name] = generator.literal(value, attr['type'])
    template = workflow / 'parm/streams.atmosphere'
    text = template.read_text()
    for name, value in request['stream_variables'].items():
        token = '@' + name + '@'
        if token not in text:
            raise ValueError('workflow stream variable is absent: ' + name)
        text = text.replace(token, str(value))
    if '@' in text:
        raise ValueError('unresolved workflow stream variable')
    streams = ET.fromstring(text)
    names = {stream.get('name') for stream in streams}
    if not set(request['streams']).issubset(names):
        raise ValueError('overlay names an absent workflow stream')
    out.mkdir(parents=True, exist_ok=False)
    deck = out / source_deck.name
    deck.mkdir()
    for stream in streams:
        for name, value in request['streams'].get(stream.get('name'), {}).items():
            if value is None:
                stream.attrib.pop(name, None)
            else:
                stream.set(name, str(value))
        for listing in stream.findall('file'):
            path = Path(listing.get('name'))
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe workflow stream list')
            target = deck / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workflow / request['stream_list_directory'] / path.name, target)
    ET.ElementTree(streams).write(deck / 'streams.atmosphere', encoding='unicode')
    (deck / 'namelist.atmosphere').write_text(''.join('&' + group + '\n' +
        ''.join('    ' + name + ' = ' + value + '\n' for name, value in values.items()) + '/\n'
        for group, values in namelist.items()))
    row['resolved_settings'].update(request['settings'])
    row['files'] = {str(path.relative_to(deck)): generator.f0.sha(path)
                    for path in sorted(deck.rglob('*')) if path.is_file()}
    index.update(decks=[row], runtime=dict(index['runtime'], workflow_overlay=request),
        overlay_authority=dict(source_index_sha256=generator.f0.sha(index_path),
            original_files=next(r for r in json.loads(index_path.read_text())['decks']
                if r['deck'] == source_deck.name)['files'],
            workflow_stream_template_sha256=generator.f0.sha(template)))
    for setting, clock in (('config_start_time', 'start'), ('config_run_duration', 'duration'),
                           ('config_dt', 'dt'), ('config_number_of_sub_steps', 'substeps')):
        if setting in request['settings']:
            index['runtime'][clock] = request['settings'][setting]
    (out / 'INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    return deck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-deck', 'workflow', 'source', 'out', 'request'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.source_deck, args.workflow, args.source, args.out,
                   json.loads(args.request.read_text())))
