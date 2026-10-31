#!/usr/bin/env python3
"""Publish the fork's public GSL CI inputs without decoding weather fields.

Authority: testing_and_setup/ufs-community/data/get_data.sh at 1d6e370e.
Download with f0 download_inputs.py first; URLs and byte hashes are retained.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha


def cache(downloads, source, work, reference):
    manifest = json.loads((downloads / 'MANIFEST.json').read_text())
    rows = {row['name']: row for row in manifest['files']}
    work.mkdir(parents=True, exist_ok=False)
    for name, target in [('mpas_data.tar', 'ics'), ('tempo_data.tar', 'tables/tempo'),
                         ('ugw_data.tar', 'tables/ugw')]:
        archive = downloads / name
        if sha(archive) != rows[name]['sha256']:
            raise ValueError(f'archive changed: {name}')
        with tarfile.open(archive) as stream:
            # Apple resource forks are archive metadata, not model inputs.
            members = [member for member in stream if not Path(member.name).name.startswith('._')]
            stream.extractall(work / target, members=members, filter='data')
    for case in ('gfs.winter', 'rap.summer', 'rap.winter'):
        inputs = work / 'ics' / ('ufscommunity.' + case)
        for name in ('mpas.init.nc', 'mpas.lbc.nc', 'mpas.ugwp_oro_data.nc'):
            if not (inputs / name).is_file():
                raise ValueError(f'missing CI input {case}/{name}')
        decks = source / 'testing_and_setup/ufs-community/cases' / ('ufscommunity.hrrrv5.' + case)
        shutil.copytree(decks, work / 'decks' / decks.name)
    (work / 'DOWNLOADS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = publish(work, reference / 'cases' / 'gsl-ci-public-v1', ['**/*'], reference / 'READY.md',
                     'Pinned fork GSL CI inputs and original hrrrv5 decks; public release tables labelled downloaded, '
                     'not rebuilt strict; run original deck with native atmosphere_model and tables linked into cwd')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('downloads', 'source', 'work', 'reference'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(cache(a.downloads, a.source, a.work, a.reference)))
