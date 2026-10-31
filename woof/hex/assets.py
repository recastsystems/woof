"""Pinned on-demand assets, independent of package-data kernel packs.

Only the installed manifest supplies pins. Offline directories and mirrors supply
bytes, verified before atomic publication under a per-asset process lock.
Run ``python -m woof.hex.assets pin --from DIR`` to regenerate the manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request

SCHEMA = 'gpuwm-hex.on-demand-assets/v1'
PINS = Path(__file__).resolve().parent / 'data/on-demand-assets.json'
FROM_ENV = 'WOOF_HEX_ASSETS_FROM'
URL_ENV = 'WOOF_HEX_ASSETS_URL'
FIRST_USE_ENV = 'WOOF_HEX_ASSETS_FIRST_USE'
#: Read-only asset directories (os.pathsep list) searched after the run's own cache: a box image stages packs there.
SYSTEM_ENV = 'WOOF_HEX_SYSTEM_ASSETS'
SYSTEM_DEFAULT = '/opt/gpuwm-hex/assets'
_VERIFIED = set()
_ATTEMPTS = {}
_RUNTIME_LOCK = threading.RLock()
_INSPECTION = ContextVar('hex_assets_inspection', default=False)


class AssetError(ValueError):
    """An asset is missing or differs from its installed pin."""


def load_pins():
    pins = json.loads(PINS.read_text())
    if pins.get('schema') != SCHEMA or not isinstance(pins.get('assets'), dict):
        raise AssetError(f'{PINS}: expected {SCHEMA}')
    for name, row in pins['assets'].items():
        if (not name or Path(name).name != name or name in ('.', '..') or '\\' in name
                or not isinstance(row.get('bytes'), int) or row['bytes'] < 0
                or len(row.get('sha256', '')) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise AssetError(f'{PINS}: invalid pin for {name!r}')
    return pins


def cache_root(cache=None):
    """Explicit --cache, prep cache root, or the per-user default when off."""
    if cache is not None:
        return Path(cache).expanduser()
    from .prep_cache import cache_root as prep_root
    root = prep_root()
    if root is not None:
        return root
    # Same per-user policy as mpasphys.resources_build.default_cache().
    declared = os.environ.get('WOOF_HEX_CACHE_ROOT')
    if declared:
        return Path(declared).expanduser()
    base = os.environ.get('XDG_CACHE_HOME')
    return (Path(base).expanduser() if base else Path.home() / '.cache') / 'woof hex'


def asset_path(name, row, cache=None):
    return cache_root(cache) / 'assets' / row['sha256'] / name


def verify(path, row, *, memoise=False):
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path.resolve()), row['sha256'], row['bytes'], stat.st_size,
               stat.st_mtime_ns, stat.st_ctime_ns)
        if memoise and key in _VERIFIED:
            return path
        if not path.is_file() or stat.st_size != row['bytes']:
            raise AssetError(f'{path}: bytes mismatch (expected {row["bytes"]}, got {stat.st_size})')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != row['sha256']:
            raise AssetError(f'{path}: sha256 mismatch (expected {row["sha256"]}, got {digest})')
    except OSError as error:
        raise AssetError(f'{path}: {error}') from error
    if memoise:
        _VERIFIED.add(key)
    return path


@contextmanager
def _lock(path):
    from .file_lock import held
    with held(path):
        yield


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def asset_url(name, pins):
    release = pins['release_assets']
    # A pack added after its siblings' asset release names its own release.
    tag = (pins.get('assets', {}).get(name) or {}).get('release_tag') or release['tag']
    base = os.environ.get(URL_ENV) or (
        f'https://github.com/{release["repository"]}/releases/download/{tag}')
    return base.rstrip('/') + '/' + name


def _download(name, row, source, origin, destination):
    with Path(destination).open('wb') as dst:
        if source is not None:
            with (Path(source) / name).open('rb') as src:
                shutil.copyfileobj(src, dst, 1 << 20)
        else:
            with urllib.request.urlopen(origin['url'], timeout=20) as src:
                shutil.copyfileobj(src, dst, 1 << 20)
    verify(destination, row)


def fetch_one(name, pins, *, cache=None, from_directory=None, _staged=None):
    """Fetch one pinned file and receipt; writers of other assets run in parallel."""
    started = time.perf_counter()
    row = pins['assets'][name]
    target = asset_path(name, row, cache)
    source = from_directory if from_directory is not None else os.environ.get(FROM_ENV)
    origin = {'source_dir': str(Path(source).resolve())} if source is not None else {'url': asset_url(name, pins)}
    receipt_path = cache_root(cache) / 'assets/receipts' / (name + '.json')
    # Bad or unreachable sources must not create even a lock in the cache.
    # A batch supplies already verified temporary bytes; first use stages here.
    with tempfile.TemporaryDirectory(prefix='hex-assets-') as staging:
        try:
            verify(target, row, memoise=True)
            needed = False
        except AssetError:
            needed = True
        if not needed and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            return {'name': name, 'path': str(target), 'status': 'staged', **receipt}
        prepared = _staged
        if needed:
            if prepared is None:
                prepared = Path(staging) / name
                _download(name, row, source, origin, prepared)
            verify(prepared, row)
        with _lock(target.parent / '.fetch.lock'):
            try:
                verify(target, row, memoise=True)
                reused = True
            except AssetError:
                reused = False
                fd, temporary = tempfile.mkstemp(prefix='.partial-', dir=target.parent)
                try:
                    with os.fdopen(fd, 'wb') as dst, Path(prepared).open('rb') as src:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    verify(temporary, row)
                    os.replace(temporary, target)
                    verify(target, row, memoise=True)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            receipt = {**origin, 'bytes': row['bytes'], 'sha256': row['sha256'],
                       'fetched_utc': datetime.now(timezone.utc).isoformat(),
                       'wall_s': time.perf_counter() - started}
            if not reused or not receipt_path.is_file():
                _atomic_json(receipt_path, receipt)
            else:
                receipt = json.loads(receipt_path.read_text())
    return {'name': name, 'path': str(target), 'status': 'staged', **receipt}


@contextmanager
def inspection_only():
    """Resolve staged bytes without copying, fetching or warning."""
    token = _INSPECTION.set(True)
    try:
        yield
    finally:
        _INSPECTION.reset(token)


def is_inspection():
    # Respect the supported engine's inspection scope without importing it.
    bridges = sys.modules.get('woof.bridges')
    return _INSPECTION.get() or bool(getattr(bridges, '_INSPECTION_ONLY', False))


def box_asset(name, row):
    """A verified copy of a pinned asset the box already holds outside this run's cache, else None.

    THE BREAKAGE THIS ENDS (an RTX 5090 host and V8X6000, 2026-10-29): packs staged once on a box (the default cache,
    or baked into its image) were invisible to a run with its own cache root, which then fetched an
    unpublished tag, got a 404 and compiled every kernel through NVRTC.  Searched: the image directories
    (WOOF_HEX_SYSTEM_ASSETS, default /opt/gpuwm-hex/assets) and the box's default cache.  Each holds
    ``<sha256>/<name>`` or ``<name>``; only bytes matching the installed pin are used.
    """
    declared = os.environ.get(SYSTEM_ENV)
    directories = [Path(part).expanduser() for part in (declared.split(os.pathsep) if declared is not None
                                                         else [SYSTEM_DEFAULT]) if part]
    base = os.environ.get('XDG_CACHE_HOME')
    directories.append((Path(base).expanduser() if base else Path.home() / '.cache') / 'woof hex/prep/assets')
    for directory in directories:
        for candidate in (directory / row['sha256'] / name, directory / name):
            if candidate.is_file():
                try:
                    return verify(candidate, row, memoise=True)
                except AssetError:
                    continue
    return None


def checkout_asset(name):
    # <checkout>/tree/src/hexcore -> <checkout>/tree/release-assets/kernel-packs.
    package = Path(__file__).resolve().parent
    for parent in package.parents:
        path = parent / 'release-assets/kernel-packs' / name
        if path.is_file():
            return path
    return None


def resolve_pack(identity):
    """Return (verified path, None), or (None, a concrete compile-path reason)."""
    with _RUNTIME_LOCK:
        try:
            pins = load_pins()
            version = identity.get('nvrtc') or ()
            major = version[0] if version else None
            candidates = [(n, r) for n, r in pins['assets'].items()
                          if r.get('kind') == 'kernel-pack' and r.get('select', {}).get('nvrtc_major') == major]
            if not candidates:
                return None, f'no kernel-pack pin for loaded NVRTC major {major}'
            from woof.hex.kernel_pack import same_compiler
            compatible = [(n, r) for n, r in candidates if same_compiler(r.get('compiler'), identity)]
            if not compatible:
                loaded = '.'.join(str(part) for part in version)
                pinned = ', '.join(sorted({'.'.join(str(p) for p in r.get('compiler', {}).get('nvrtc', ()))
                                           + ' (' + str(r.get('compiler', {}).get('nvrtc_package')) + ')'
                                           for _, r in candidates}))
                return None, (f'loaded NVRTC {loaded} (library sha256 '
                              f'{str(identity.get("nvrtc_library_sha256"))[:12]}) matches no NVRTC {major} pin '
                              f'[{pinned}]; no compatible pack is downloaded')
            name, row = compatible[0]
            reason = 'not staged'
            target = asset_path(name, row)
            if target.exists():
                try:
                    return verify(target, row, memoise=True), None
                except AssetError as error:
                    reason = str(error)
            held = box_asset(name, row)
            if held is not None:
                return held, None
            local = checkout_asset(name)
            if local is not None:
                try:
                    return verify(local, row, memoise=True), None
                except AssetError as error:
                    reason = str(error)
            if is_inspection():
                return None, 'inspection only; no fetch'
            key = (name, row['sha256'])
            if key in _ATTEMPTS:
                return None, _ATTEMPTS[key]
            source = os.environ.get(FROM_ENV)
            if source is None and os.environ.get(FIRST_USE_ENV, '').lower() == 'off':
                return None, f'{reason}; first-use off ({FIRST_USE_ENV}=off)'
            # Remember the attempt before I/O, including an offline copy failure.
            _ATTEMPTS[key] = reason
            try:
                result = fetch_one(name, pins, from_directory=source)
                return Path(result['path']), None
            except (OSError, ValueError) as error:
                reason = f'{name}: {"offline source" if source is not None else "fetch unavailable (offline or network failure)"}: {error}'
                _ATTEMPTS[key] = reason
                return None, reason
        except (OSError, ValueError, KeyError, TypeError) as error:
            return None, f'asset pins unavailable: {error}'


def staged_rows(pins=None, cache=None):
    """Inspect every pin without network or mutation, always rechecking bytes."""
    pins = load_pins() if pins is None else pins
    rows = []
    for name, row in pins['assets'].items():
        path = asset_path(name, row, cache)
        source = os.environ.get(FROM_ENV)
        if not path.exists() and source is not None:
            path = Path(source).expanduser() / name
        result = {'name': name, **row, 'path': str(path)}
        if not path.exists():
            result.update(status='on demand', detail=f'on demand; fetched on first use; stage with `woof hex fetch-assets --only {name}`')
        else:
            try:
                verify(path, row)
                result.update(status='staged', detail=f'staged; sha256 {row["sha256"][:12]}; {path}')
            except AssetError as error:
                result.update(status='invalid', detail=f'{error}; run `woof hex fetch-assets`')
        rows.append(result)
    return rows


def run_fetch(arguments):
    try:
        pins = load_pins()
        names = arguments.only or list(pins['assets'])
        unknown = set(names) - pins['assets'].keys()
        if unknown:
            raise AssetError('unknown assets: ' + ', '.join(sorted(unknown)))
        if arguments.list:
            results = [r for r in staged_rows(pins, arguments.cache) if r['name'] in names]
        else:
            names = list(dict.fromkeys(names))
            source = arguments.from_directory if arguments.from_directory is not None else os.environ.get(FROM_ENV)
            def prepare(name, staging):
                try:
                    row = pins['assets'][name]
                    try:
                        verify(asset_path(name, row, arguments.cache), row)
                        return None
                    except AssetError:
                        pass
                    path = Path(staging) / name
                    _download(name, row, source, {'url': asset_url(name, pins)}, path)
                    return path
                except (OSError, ValueError) as error:
                    return {'name': name, 'status': 'failed', 'detail': str(error)}
            # Verify the complete requested batch before publishing any of it.
            # A bad hash or offline mirror leaves a fresh cache entirely empty.
            with tempfile.TemporaryDirectory(prefix='hex-assets-batch-') as staging:
                with ThreadPoolExecutor(max_workers=min(8, max(2, len(names)))) as pool:
                    prepared = list(pool.map(lambda name: prepare(name, staging), names))
                    failed = [row for row in prepared if isinstance(row, dict)]
                    if failed:
                        cause = ', '.join(row['name'] for row in failed)
                        results = [row if isinstance(row, dict) else
                                   {'name': name, 'status': 'failed', 'detail': f'batch refused because {cause} could not be verified; cache unchanged'}
                                   for name, row in zip(names, prepared)]
                    else:
                        def publish(pair):
                            name, path = pair
                            try:
                                return fetch_one(name, pins, cache=arguments.cache,
                                                 from_directory=arguments.from_directory, _staged=path)
                            except (OSError, ValueError) as error:
                                return {'name': name, 'status': 'failed', 'detail': str(error)}
                        results = list(pool.map(publish, zip(names, prepared)))
        if arguments.json:
            print(json.dumps({'schema': SCHEMA, 'assets': results}, indent=2, sort_keys=True))
        else:
            for row in results:
                if row['status'] == 'failed':
                    print(f'{row["name"]}: failed: {row["detail"]}', file=sys.stderr)
                elif arguments.list:
                    print(f'{row["name"]} {row["bytes"]} B {row["sha256"][:12]} {row["detail"]}')
                else:
                    print(f'{row["name"]} {row["bytes"]} B {row["sha256"][:12]} '
                          f'{row.get("source_dir", row.get("url"))} {row["wall_s"]:.3f} s')
        return int(any(r['status'] in ('failed', 'invalid') for r in results))
    except (OSError, ValueError, KeyError) as error:
        print(f'woof hex fetch-assets: {error}', file=sys.stderr)
        return 1


def add_fetch_assets_parser(commands):
    parser = commands.add_parser('fetch-assets', help='stage verified on-demand assets from the release or an offline directory')
    parser.add_argument('--from', dest='from_directory', type=Path, metavar='DIR', help='offline source of bytes; pins come from this installation')
    parser.add_argument('--cache', type=Path, metavar='DIR', help='cache root (assets are stored under DIR/assets)')
    parser.add_argument('--only', nargs='+', metavar='NAME', help='fetch only these pinned assets (default: all)')
    parser.add_argument('--list', action='store_true', help='show pins and staged state without fetching')
    parser.add_argument('--json', action='store_true', help='emit the results as JSON')
    parser.set_defaults(handler=run_fetch)
    return parser


def pin(arguments):
    """Pin every ``kernel-pack-cuda<N>[.<minor>].json`` in the directory.

    One pack per loaded NVRTC: CUDA 12.9 pip, CUDA 13.4 pip, and CUDA 13.0
    (the system toolkit of a driver-580 machine and the 13.0.88 pip wheel
    carry the same library bytes, measured 2026-10-24).
    """
    assets = {}
    names = sorted(path.name for path in Path(arguments.from_directory).glob('kernel-pack-cuda*.json'))
    for name in names:
        match = re.fullmatch(r'kernel-pack-cuda(\d+)(?:\.(\d+))?\.json', name)
        if match is None:
            raise AssetError(f'{name}: not a kernel-pack-cuda<N>[.<minor>].json name')
        major = int(match.group(1))
        data = (arguments.from_directory / name).read_bytes()
        compiler = json.loads(data)['compiler']
        if compiler['nvrtc'][0] != major or len(compiler) != 9:
            raise AssetError(f'{name}: expected the CUDA {major} nine-key compiler identity')
        if match.group(2) is not None and compiler['nvrtc'][1] != int(match.group(2)):
            raise AssetError(f'{name}: built with NVRTC {compiler["nvrtc"]}, not {major}.{match.group(2)}')
        assets[name] = {'kind': 'kernel-pack', 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(),
                        'select': {'nvrtc_major': major}, 'compiler': compiler}
    if not assets:
        raise AssetError(f'{arguments.from_directory}: no kernel-pack-cuda*.json files')
    document = {'schema': SCHEMA,
                'breakage': 'The 169 MB woof hex 0.4.1 wheel exceeds the PyPI 100,000,000 B per-file cap.',
                'release_assets': {'repository': arguments.repository, 'tag': arguments.tag}, 'assets': assets}
    _atomic_json(arguments.output, document)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m woof.hex.assets')
    commands = parser.add_subparsers(required=True)
    add_fetch_assets_parser(commands)
    generator = commands.add_parser('pin', help='regenerate pins from the exact release-asset bytes')
    generator.add_argument('--from', dest='from_directory', required=True, type=Path)
    generator.add_argument('--output', type=Path, default=PINS)
    generator.add_argument('--repository', default='FahrenheitResearch/gpuwm-hex')
    generator.add_argument('--tag', default='kernel-packs-v0.4.9')
    generator.set_defaults(handler=pin)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
