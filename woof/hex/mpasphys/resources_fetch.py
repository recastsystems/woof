"""Fetch the release's DATA and assemble compiler-free MPAS resources.

Only the packaged manifest supplies hashes. An offline directory supplies bytes,
not replacement pins. Archives contain regular files, extracted and verified in
parallel into a private directory before an atomic publication.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target, image_target as _image_target

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile

from . import resources_build as rb

SCHEMA = 'gpuwm-hex.physics-resource-assets/v1'
MANIFEST = Path(__file__).resolve().parents[1] / 'data/physics-resource-assets.json'
FROM_ENV = 'WOOF_HEX_PHYSICS_RESOURCES_FROM'
URL_ENV = 'WOOF_HEX_PHYSICS_RESOURCES_URL'


def load_manifest():
    try:
        raw = MANIFEST.read_bytes()
    except FileNotFoundError as error:
        raise rb.PhysicsResourceError(f'{MANIFEST}: this installation has no sealed physics release manifest; '
                                      'install a release wheel or explicitly build from source') from error
    pins = json.loads(raw)
    if pins.get('schema') != SCHEMA or not pins.get('assets'):
        raise rb.PhysicsResourceError(f'{MANIFEST}: no sealed physics release assets; use an explicit '
                                      '`woof hex physics-resources build` from source')
    return pins, hashlib.sha256(raw).hexdigest()


def _relative(value):
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or '..' in path.parts or '\\' in value or str(path) != value:
        raise rb.PhysicsResourceError(f'unsafe physics asset path {value!r}')
    return Path(*path.parts)


@contextmanager
def _lock(cache):
    from ..file_lock import held
    with held(cache / '.fetch.lock'):
        yield


def _check_file(path, row):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']
            or rb.sha256_file(path) != row['sha256']):
        raise rb.PhysicsResourceError(f'physics release file {path} is absent or differs from pinned '
                                      f"sha256 {row['sha256']}")


def _download(asset, cache, source, release):
    name = asset['name']
    if _relative(name).name != name:
        raise rb.PhysicsResourceError(f'physics release asset name must be a basename: {name!r}')
    target = cache / 'downloads' / asset['sha256'] / name
    if target.is_file():
        try:
            _check_file(target, asset)
            return target
        except rb.PhysicsResourceError:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix='.partial-')
    try:
        with os.fdopen(descriptor, 'wb') as dst:
            if source is not None:
                local = Path(source) / name
                if not local.is_file():
                    raise rb.PhysicsResourceError(f'offline physics asset {local} is missing')
                with local.open('rb') as src:
                    shutil.copyfileobj(src, dst, 1 << 20)
            else:
                base = os.environ.get(URL_ENV)
                url = (base.rstrip('/') + '/' + name if base else asset.get('url') or
                       f"https://github.com/{release['repository']}/releases/download/{release['tag']}/{name}")
                with urllib.request.urlopen(url, timeout=300) as src:
                    shutil.copyfileobj(src, dst, 1 << 20)
        _check_file(Path(temporary), asset)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _extract_member(archive, info, row, directory):
    relative = _relative(info.filename)
    mode = info.external_attr >> 16
    if info.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
        raise rb.PhysicsResourceError(f'physics asset has a non-regular member: {info.filename}')
    if mode & 0o111:
        raise rb.PhysicsResourceError(f'physics asset has an executable member: {info.filename}')
    if info.file_size != row['bytes']:
        raise rb.PhysicsResourceError(f'physics asset member size differs: {info.filename}')
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as src, path.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    _check_file(path, row)
    path.chmod(0o644)


def _install_upstream(source, row, directory):
    # Offline mirrors carry the original upstream file beside the release ZIPs.
    # It is never embedded in a release archive, including duplicate tree paths.
    for name in row['paths']:
        path = directory / _relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, path.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        _check_file(path, row)
        path.chmod(0o644)


def _verify(directory, assets, upstream, jobs):
    members = [(directory / _relative(name), row)
               for asset in assets for name, row in asset['files'].items()]
    members.extend((directory / _relative(name), row) for row in upstream for name in row['paths'])
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda pair: _check_file(*pair), members))


def verify_installed_data(cache=None):
    """Read and hash installed MPAS DATA without fetching or changing its cache."""
    pins, digest = load_manifest()
    record = rb.read_installed(cache)
    row = record['resources']['thompson-tables']
    if row.get('release_manifest_sha256') != digest:
        raise rb.PhysicsResourceError('installed MPAS DATA belongs to another release manifest; '
                                      'run `woof hex physics-resources fetch`')
    path = rb.resolve_resource('thompson-tables', cache)
    relative = _relative(pins['resources']['thompson-tables']['path'])
    directory = path
    for _ in relative.parts:
        directory = directory.parent
    assets = [row for row in pins['assets'] if 'v8.4.2' in row['pin_sets']]
    upstream = [row for row in pins.get('upstream_files', []) if 'v8.4.2' in row['pin_sets']]
    _verify(directory, assets, upstream, 8)
    return {'path': str(directory), 'manifest_sha256': digest,
            'verified_files': sum(len(row['files']) for row in assets)
                              + sum(len(row['paths']) for row in upstream)}


def fetch(cache=None, *, from_directory=None, pin_sets=rb.PIN_SETS, jobs=None,
          log=rb._log):
    """Install sealed resources without invoking a compiler or a source builder."""
    started = time.perf_counter()
    pins, digest = load_manifest()
    pin_sets = tuple(dict.fromkeys(pin_sets))
    unknown = set(pin_sets) - set(rb.PIN_SETS)
    if unknown or not pin_sets:
        raise rb.PhysicsResourceError(f'unknown physics pin sets: {sorted(unknown)}')
    if jobs is None:
        jobs = max(2, min(8, os.cpu_count() or 2))
    if jobs < 2 or jobs > 64:
        raise rb.PhysicsResourceError('physics resource fetch --jobs must be between 2 and 64')
    cache = Path(cache if cache is not None else rb.default_cache()).resolve()
    source = from_directory if from_directory is not None else os.environ.get(FROM_ENV)
    assets = [row for row in pins['assets'] if set(row['pin_sets']) & set(pin_sets)]
    upstream = [row for row in pins.get('upstream_files', []) if set(row['pin_sets']) & set(pin_sets)]
    # Releases carry DATA only. Helpers, Registry metadata and device maths
    # come from the installed wheel, never from an asset archive.
    if set(pins['resources']) - {'thompson-tables'}:
        raise rb.PhysicsResourceError('physics release manifest must contain DATA resources only')
    for asset in assets:
        for name, row in asset['files'].items():
            if row.get('executable') or Path(name).suffix not in ('.DBL', '.bin'):
                raise rb.PhysicsResourceError('physics release asset must contain non-executable DATA only: ' + name)
    if not assets:
        raise rb.PhysicsResourceError('the packaged physics manifest has no assets for these pin sets')
    names = [name for asset in assets for name in asset['files']]
    names.extend(name for row in upstream for name in row['paths'])
    for name in names:
        _relative(name)
    if len(names) != len(set(names)):
        raise rb.PhysicsResourceError('physics release assets overlap file paths')
    # A pin-set-specific directory avoids replacing a live installation as another
    # process fetches the other fork. All packet paths stay relative within it.
    selection = '-'.join(sorted(pin_sets))
    destination = cache / 'releases' / digest / selection
    timings = {}
    with _lock(cache):
        if destination.exists():
            then = time.perf_counter()
            _verify(destination, assets, upstream, jobs)
            timings['verify_seconds'] = time.perf_counter()-then
        else:
            then = time.perf_counter()
            log(f'fetching {len(assets)} pinned physics assets and {len(upstream)} upstream files with {jobs} workers'
                + (f' from {source}' if source is not None else ' from the release'))
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                downloads = list(pool.map(lambda asset: _download(asset, cache, source, pins['release_assets']),
                                          [*assets, *upstream]))
            archives = downloads[:len(assets)]
            timings['download_verify_seconds'] = time.perf_counter()-then
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.partial-', dir=destination.parent))
            opened = []
            try:
                then = time.perf_counter()
                work = []
                for asset, path in zip(assets, archives):
                    archive = zipfile.ZipFile(path)
                    opened.append(archive)
                    infos = archive.infolist()
                    if len(infos) != len(asset['files']) or {i.filename for i in infos} != set(asset['files']):
                        raise rb.PhysicsResourceError(f'{path}: archive members differ from the packaged manifest')
                    for info in infos:
                        work.append((archive, info, asset['files'][info.filename], temporary))
                # zlib decompression releases the GIL. Each independent ZIP member
                # is read, expanded and hashed by a worker, without a serial tar stage.
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    futures = [pool.submit(_extract_member, *args) for args in work]
                    futures.extend(pool.submit(_install_upstream, path, row, temporary)
                                   for row, path in zip(upstream, downloads[len(assets):]))
                    for future in futures:
                        future.result()
                os.replace(temporary, destination)
                timings['parallel_extract_verify_seconds'] = time.perf_counter()-then
            finally:
                for archive in opened:
                    archive.close()
                if temporary.exists():
                    shutil.rmtree(temporary)
        rows = {}
        for key, row in pins['resources'].items():
            if not set(row['pin_sets']) & set(pin_sets):
                continue
            value = {k:v for k,v in row.items() if k != 'pin_sets'}
            value['path'] = str(destination / _relative(value['path']))
            value['release_manifest_sha256'] = digest
            if 'variants' in value:
                value['variants'] = {mode:{arch:{**item, 'path':str(destination/_relative(item['path']))}
                                          for arch,item in architectures.items()}
                                     for mode,architectures in value['variants'].items()}
            rows[key] = value
        then = time.perf_counter()
        rows.update(rb.assemble_framework_resources(destination, log=log))
        timings['wheel_resource_assembly_seconds'] = time.perf_counter()-then
        installed = rb.record_installed(cache, rows)
        # Expanded resources are the cache. Retaining compressed copies doubles
        # disk use without helping a subsequent verified installation.
        for asset in [*assets, *upstream]:
            archive = cache/'downloads'/asset['sha256']/asset['name']
            archive.unlink(missing_ok=True)
    result = {'schema':'gpuwm-hex.physics-resources-fetch/v1','manifest_sha256':digest,
            'pin_sets':list(pin_sets),'from':str(source) if source is not None else 'release',
            'installed':str(installed),'resources':rows,'workers':jobs,'stages':timings,
            'upstream_files':len(upstream),
            'wall_seconds':time.perf_counter()-started,'source_build':False}
    # First-use callers need the same stage receipt as the explicit fetch CLI.
    receipts=cache/'receipts';receipts.mkdir(exist_ok=True)
    descriptor, temporary=tempfile.mkstemp(dir=receipts,prefix='.fetch-')
    try:
        with os.fdopen(descriptor,'w') as stream:
            stream.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
        os.replace(temporary,receipts/f'fetch-{digest}-{selection}.json')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def packet_path(cache=None, *, landuse=None):
    """Resolve an implicit packet for the run's numerics and GPU architecture."""
    from .. import numerics
    row = rb.read_installed(cache)['resources']['physics-resources']
    variants = row.get('variants')
    where = rb.installed_record_path(cache)
    if variants:
        import cupy
        arch = _image_target(xp=cupy)
        try:
            choices=variants[numerics.mode()]
            if arch.startswith('gfx') and arch not in choices:
                # Table/data payloads are host-generated and ISA independent.
                # MYNN's embedded CUDA images are authenticated then rebuilt
                # from their canonical source by load_cuda_packet on HIP.
                chosen=choices[sorted(choices)[0]]
            else:
                chosen = choices[arch]
        except KeyError as error:
            raise rb.PhysicsResourceError(f'no prebuilt physics packet for {numerics.mode()}/{arch}') from error
        path = rb._rebased(Path(chosen['path']), rb._recorded_root(rb.read_installed(cache)), where.parent.resolve())
        _check_file(path, chosen)
    else:
        path = rb.resolve_resource('physics-resources', cache)
    if landuse:
        # The decoder reads the selected LANDUSE.TBL section at run time. This
        # changes only its section name; it regenerates no native table words.
        packet = rb.load_packet(path)
        if packet['sf_noah'].get('landuse') == landuse:
            return path
        packet['sf_noah']['landuse'] = landuse
        value = json.dumps(rb.portable_packet(packet, where.parent), sort_keys=True, indent=2)+'\n'
        target = where.parent/'packets'/f"physics-resources-{hashlib.sha256(value.encode()).hexdigest()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix='.partial-')
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(value)
            os.replace(temporary, target)
        return target
    return path


def command(arguments):
    result = fetch(arguments.cache, from_directory=arguments.from_directory,
                   pin_sets=tuple(arguments.pin_set or ('v8.4.2',)), jobs=arguments.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def add_parser(verbs):
    parser = verbs.add_parser('fetch', help='fetch DATA and assemble wheel resources; no compiler needed')
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--from', dest='from_directory', type=Path, default=None,
                        help='offline directory containing release ZIPs and the pinned upstream table files')
    parser.add_argument('--pin-set', action='append', choices=rb.PIN_SETS, default=None)
    parser.add_argument('--jobs', type=int, default=None, help='parallel download/extraction workers (default: up to 8)')
    parser.set_defaults(handler=command)
