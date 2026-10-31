#!/usr/bin/env python3
"""Index pristine source decks verbatim for the same native forecast runner."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rrfs_generated_decks', HERE / 'generate_decks.py')
decks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decks)


def catalog(source, rows, out):
    groups, registries = decks.registry(source)
    _, suites = decks.f0.options(source)
    entries = []
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        original = Path(row['source']).resolve(strict=True)
        target = out / row['name']
        target.mkdir()
        for path in sorted(original.iterdir()):
            if path.is_file():
                shutil.copy2(path, target / path.name)
        raw = decks.read_namelist((target / 'namelist.atmosphere').read_text())
        explicit = {key: decks.decoded(value, groups[key][1]['type'])
                    for record in raw.values() for key, value in record.items() if key in groups}
        suite = explicit['config_physics_suite']
        resolved = dict(suites[suite])
        resolved.update(explicit)
        entries.append(dict(deck=row['name'], suite=suite, resolved_suite=suites[suite],
                            override=explicit, companions={}, resolved_settings=resolved,
                            source=str(original), source_kind='pristine-source-deck-verbatim',
                            unregistered_options={key: value for record in raw.values()
                                                  for key, value in record.items() if key not in groups},
                            files={p.name: decks.f0.sha(p) for p in sorted(target.iterdir()) if p.is_file()}))
    index = dict(schema='mpas.rrfs-source-decks/v1', runtime={'source_rows': rows}, decks=entries,
                 source_sha256={str(p.relative_to(source)): decks.f0.sha(p)
                                for p in sorted(registries | {source / decks.f0.CONTROL})})
    (out / 'INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({'decks': len(catalog(a.source, json.loads(a.index.read_text())['decks'], a.out)['decks'])}))
