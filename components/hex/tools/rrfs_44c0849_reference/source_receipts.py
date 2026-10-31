#!/usr/bin/env python3
"""Record checked-out public pins and plain per-area source deltas."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=120).strip()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + '\n')


def ready(root, label, artifact, use):
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with (root / 'READY.md').open('a') as f:
        f.write(f'{label} | {artifact} | sha256 {digest} | {use}\n')


def main(root):
    src = root / 'src/MPAS-Model'
    wf = root / 'wf/rrfs-workflow'
    new = git(src, 'rev-parse', 'HEAD')
    old = git(src, 'rev-parse', '1d6e370e')
    links = lambda repo, rev: {line.split()[3]: line.split()[2] for line in git(repo, 'ls-tree', '-r', rev).splitlines() if line.startswith('160000 ')}
    modules = []
    for path, commit in links(src, new).items():
        assert git(src / path, 'rev-parse', 'HEAD') == commit
        modules.append(dict(path=path, commit=commit, url=git(src / path, 'remote', 'get-url', 'origin'), describe=git(src / path, 'describe', '--tags', '--always', '--dirty')))
    pins = dict(model_commit=new, model_url='https://github.com/RRFSx/MPAS-Model', readme_version=(src / 'README.md').read_text().splitlines()[0], submodules=modules,
                workflow_commit=git(wf, 'rev-parse', 'HEAD'), workflow_url='https://github.com/NOAA-EMC/rrfs-workflow', workflow_version=(wf / 'workflow/VERSION').read_text().strip(), workflow_gitlinks=links(wf, 'HEAD'))
    write(root / 'SOURCE-PINS.json', json.dumps(pins, indent=2))
    ready(root, 'Native source and workflow pins', root / 'SOURCE-PINS.json', 'Read src/MPAS-Model and wf/rrfs-workflow; source checkouts are immutable inputs.')
    delta = root / 'delta'
    write(delta / 'old-to-new.stat.txt', git(src, 'diff', '--stat', old, new))
    counts = git(src, 'diff', '--numstat', old, new)
    write(delta / 'old-to-new.numstat.tsv', counts)
    for line in counts.splitlines():
        plus, minus, path = line.split('\t')
        if not path.startswith(('src/core_atmosphere/', 'src/core_init_atmosphere/')):
            continue
        area = 'init' if path.startswith('src/core_init') else 'dycore' if '/dynamics/' in path or '/mpas_atm_core' in path or '/mpas_atm_halos' in path else 'physics-registry-streams'
        write(delta / area / (path.replace('/', '__') + '.diff'), git(src, 'diff', old, new, '--', path))
    write(delta / 'upstream-v8.4.1-to-new.stat.txt', git(src, 'diff', '--stat', 'v8.4.1', new))
    write(delta / 'upstream-v8.4.1-to-new.numstat.tsv', git(src, 'diff', '--numstat', 'v8.4.1', new))
    write(delta / 'dycore/upstream-v8.4.1-to-new.diff', git(src, 'diff', 'v8.4.1', new, '--', 'src/core_atmosphere/dynamics', 'src/core_atmosphere/mpas_atm_core.F', 'src/core_atmosphere/mpas_atm_core_interface.F', 'src/core_atmosphere/mpas_atm_halos.F'))
    oldlinks = links(src, old)
    for module in modules:
        path, commit = module['path'], module['commit']
        name = Path(path).name
        if name in ('MYNN-EDMF', 'SMOKE', 'TEMPO'):
            before = oldlinks[path]
            write(delta / 'submodules' / (name + '.diff'), f'# {before}..{commit}\n' + git(src / path, 'diff', before, commit))
            write(delta / 'submodules' / (name + '.stat.txt'), git(src / path, 'diff', '--stat', before, commit) or 'No change: identical gitlink.')
        if name == 'stochastic_physics':
            target = delta / 'stochastic_physics'
            target.mkdir(exist_ok=True)
            import tarfile
            proc = subprocess.Popen(['git', '-C', str(src / path), 'archive', commit], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
                archive.extractall(target, filter='data')
            proc.stdout.close()
            assert proc.wait(timeout=120) == 0
    files = {str(p.relative_to(delta)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(delta.rglob('*')) if p.is_file()}
    write(delta / 'MANIFEST.json', json.dumps(dict(old=old, new=new, files=files), indent=2))
    ready(root, 'Source deltas by area', delta / 'MANIFEST.json', 'Per-file patches in dycore/, physics-registry-streams/, init/; submodule patches and full stochastic_physics/ tree.')
    next_rows = git(src, 'log', 'refs/remotes/ufs/noaa-develop', '--format=%H %s', '--grep=v8.4.2-1.20', '-1')
    assert next_rows, 'next pin title not found'
    pin = next_rows.split()[0]
    write(root / 'NEXT-PIN.md', '# Next pin, record only\n\n' + git(src, 'show', '-s', '--format=%H%n%aI%n%s', pin) + '\n\n```text\n' + git(src, 'diff', '--stat', new, pin) + '\n```')
    ready(root, 'Next pin note', root / 'NEXT-PIN.md', 'Recorded only; not built or ported.')
    print(json.dumps(dict(model=new, workflow=pins['workflow_commit'], submodules=len(modules), delta_files=len(files), next_pin=pin)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root.resolve())
