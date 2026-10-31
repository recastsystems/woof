"""One immutable static per mesh, builder, geography and option set per host.

THE BREAKAGE THIS PREVENTS: nothing a static contains depends on the forecast
date or the member, yet every ensemble preparation and every mesh-point run
rebuilt the same static (73-206 s each, same bytes) on the same host.  Here a
static is built once under a content key and every consumer links to that one
artifact; a verified rebuild (``verify_rebuild``) compares whole-file bytes.

The key: the mesh file's SHA-256, the builder executable's SHA-256 (its
version, lake corrections included), every geography file's name and SHA-256,
and the content options.  File digests are reused only for an unchanged file
version; geography names are inventoried on every call, so adding or removing
a tile invalidates the key.  Publication is atomic and locked.  Existing
unverified outputs are never adopted.

Where the cache is (``cache_dir``): ``WOOF_HEX_STATIC_CACHE``, else
``$WOOF_HEX_CACHE_ROOT/static``, else ``~/.cache/gpuwm-hex/static``.  The
mesh-point door uses it only when one of the two variables is set
(``door_cache_dir``), so a bare install still writes its statics in place;
``WOOF_HEX_STATIC_CACHE=off`` turns the door's use off.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import sys
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

SCHEMA = "hex.static-cache/v1"
ENV_CACHE = "WOOF_HEX_STATIC_CACHE"
ENV_ROOT = "WOOF_HEX_CACHE_ROOT"
_OFF = ("off", "0", "none")


def cache_dir(environ=None):
    """The static cache directory for a caller that asked for the cache.

    ``WOOF_HEX_STATIC_CACHE``, else ``$WOOF_HEX_CACHE_ROOT/static``, else
    ``~/.cache/gpuwm-hex/static``.  ``WOOF_HEX_STATIC_CACHE=off`` with no
    explicit ``cache=`` is a refusal: the caller asked for a cached static and
    the environment says there is no cache, and building in place instead
    would silently drop the one-static-per-host guarantee the caller relies on.
    """
    environ = os.environ if environ is None else environ
    text = environ.get(ENV_CACHE, "")
    if text.lower() in _OFF:
        raise ValueError(f"{ENV_CACHE}={text} turns the static cache off, but build_static was called without an "
                         f"explicit cache=; pass cache=DIR, or unset {ENV_CACHE}")
    if text:
        return Path(text).expanduser().absolute()
    root = environ.get(ENV_ROOT)
    return (Path(root).expanduser() if root else Path.home() / ".cache" / "woof hex") / "static"


def door_cache_dir(environ=None):
    """The cache a door uses, or None to build the static in place (the door's behaviour before this cache).

    On only when ``WOOF_HEX_STATIC_CACHE`` names a directory or ``WOOF_HEX_CACHE_ROOT`` is set.  A bare
    install keeps writing real files where it was asked to: a cached output is a symbolic link into the cache,
    and a link copied to another host, or left after the cache is cleared, points at nothing.
    """
    environ = os.environ if environ is None else environ
    text = environ.get(ENV_CACHE, "")
    if text.lower() in _OFF:
        return None
    if text or environ.get(ENV_ROOT):
        return cache_dir(environ)
    return None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _version(stat):
    # THE BREAKAGE THIS PREVENTS (hex fast start, 2026-10-23): a fresh box from the image
    # spent 506 s of a 744 s 1 km static re-hashing all 30.9 GB of WPS_GEOG, cold from the snapshot,
    # to form the cache key.  st_dev is the block device's number, which a new box (another instance
    # type, another NVMe enumeration) can change for the very same file system, so a memo baked into
    # the image could never hit.  The path (the memo's own key), inode, size, mtime and ctime still
    # pin one file version; with a filled memo the digest takes 1.35 s.
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _memo_entry(path, cache):
    return Path(cache) / (hashlib.sha256(str(path).encode()).hexdigest() + ".json")


def _memoized(path, version, cache):
    """The memoized hash of this exact file version, or None."""
    try:
        record = json.loads(_memo_entry(path, cache).read_bytes())
        if record['path'] == str(path) and record['version'] == version:
            return record['sha256']
    except (OSError, ValueError, KeyError):
        pass
    return None


def digest(path, cache=None):
    """Hash actual bytes, refusing a concurrent change; memoize stable versions."""
    path = Path(path).resolve(strict=True)
    before = path.stat()
    version = _version(before)
    entry = None
    if cache is not None:
        entry = _memo_entry(path, cache)
        known = _memoized(path, version, cache)
        if known is not None:
            return known
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    if _version(path.stat()) != version:
        raise RuntimeError(f"static input changed while hashing: {path}")
    result = h.hexdigest()
    # Do not memoize files still inside the filesystem timestamp tick.
    if entry is not None and time.time_ns() - before.st_mtime_ns > 2_000_000_000:
        _write(entry, _json(dict(path=str(path), version=version, sha256=result)))
    return result


def geography(root, cache):
    """Content manifest using logical names, following dataset directory links."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"geography root is not a directory: {root}")
    files = []

    def walk(folder, relative, ancestors):
        actual = folder.resolve(strict=True)
        if actual in ancestors:
            raise ValueError(f"geography symlink cycle: {folder}")
        for child in sorted(folder.iterdir()):
            name = relative / child.name
            # A hidden entry is never geography: it is left out of the manifest, unread and unhashed, so a
            # credential or config file in the tree can never reach a key or a receipt.  It used to be refused,
            # and a box kit's completion marker (`.geog-all.done`, 2026-10-24) failed every
            # hex region run on those boxes until the file was moved.  woof's own `.fetch-geog` receipts and
            # `.keep` were already skipped the same way.
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                walk(child, name, ancestors | {actual})
            elif child.is_file():
                files.append((name.as_posix(), child))
            else:
                raise ValueError(f"unsupported geography entry: {child}")

    walk(root, Path(), set())
    if not files:
        raise ValueError(f"empty geography root: {root}")
    # THE BREAKAGE THIS PREVENTS (unified-image check, 2026-10-16): the ~14,000 files of a full
    # WPS_GEOG were hashed one after another on one thread (58 s cold on the home RTX 5090 host) while every card
    # waited for the static.  Memoized versions are read in walk order here (cheap); the files that
    # need hashing go to a thread pool on every core (sha256 and reads release the GIL).  The walk
    # order and every entry are unchanged, so the manifest and its hash are the same bytes.
    manifest = []
    todo = []
    for name, child in files:
        known = None
        if cache is not None:
            actual = child.resolve(strict=True)
            known = _memoized(actual, _version(actual.stat()), cache)
        manifest.append(dict(name=name, bytes=child.stat().st_size, sha256=known))
        if known is None:
            todo.append((len(manifest) - 1, child))
    workers = max(1, min(len(todo), os.cpu_count() or 1))
    if workers == 1:
        hashes = [digest(child, cache) for _, child in todo]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geog-hash") as pool:
            hashes = list(pool.map(lambda item: digest(item[1], cache), todo))
    for (index, _), value in zip(todo, hashes):
        manifest[index]["sha256"] = value
    return dict(sha256=hashlib.sha256(_json(manifest)).hexdigest(), files=manifest)


@contextmanager
def _lock(path):
    from .file_lock import held
    with held(path, mode='a+b'):
        yield


def _options(options):
    """Only content options; never accept an extra output or geography path."""
    result = list(map(str, options))
    i = 0
    while i < len(result):
        if result[i] == '--terrain-only':
            i += 1
        elif result[i] in ('--nominal-dx-m', '--valid-time', '--supersample-30s') and i + 1 < len(result):
            i += 2
        else:
            raise ValueError(f"unsupported cached static option: {result[i]}")
    return result


def build_static(*, engine, grid, geog, out, receipt, cache=None, options=(), verify_rebuild=False, clobber=False,
                 runner=None):
    """Build once and link to verified bytes; return a separate cache receipt.

    `verify_rebuild` runs the builder again under the same key, compares whole
    file bytes, and records both hashes. It never promotes a differing rebuild.
    Incomplete/failed build folders are retained for diagnosis and owned cleanup.
    `cache`: the cache directory (default ``cache_dir()``).  `runner`: called as
    ``runner(command, log_path)`` to run the builder command (default: run it
    here, output to ``log_path``); it raises when the build fails.
    """
    started = time.monotonic()
    cache = Path(cache) if cache else cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    engine, grid, geog = (Path(p).resolve(strict=True) for p in (engine, grid, geog))
    out, receipt = Path(out), Path(receipt)
    options = _options(options)
    hashes = cache / 'digests'
    geo = geography(geog, hashes)
    material = dict(schema=SCHEMA, mesh_sha256=digest(grid, hashes),
                    builder_sha256=digest(engine, hashes), geography_sha256=geo['sha256'],
                    options=options)
    key = hashlib.sha256(_json(material)).hexdigest()
    entry = cache / key
    artifact = entry / 'static.nc'
    cache_result = dict(schema=SCHEMA, key=key, material=material, cache=str(entry))
    with _lock(cache / (key + '.lock')):
        hit = entry.is_dir()
        if hit:
            saved = json.loads((entry / 'complete.json').read_bytes())
            if saved['material'] != material or digest(artifact, hashes) != saved['sha256']:
                raise RuntimeError(f"static cache integrity failure: {entry}")
        if not hit or verify_rebuild:
            staging = cache / (key + '.building-' + uuid.uuid4().hex)
            staging.mkdir()
            output = staging / 'static.nc'
            command = [str(engine), '--grid', str(grid), '--geog', str(geog),
                       '--out', str(output), '--receipt', str(staging / 'builder.json'), *options]
            before = time.monotonic()
            if runner is None:
                with (staging / 'build.log').open('wb') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            else:
                runner(command, staging / 'build.log')
            rebuild_sha = digest(output)
            # Inputs may have changed during the build; never publish under a stale key.
            after_geo = geography(geog, hashes)
            if (digest(grid, hashes) != material['mesh_sha256'] or
                    digest(engine, hashes) != material['builder_sha256'] or after_geo != geo):
                raise RuntimeError(f"static inputs changed during build: {staging}")
            record = dict(material=material, sha256=rebuild_sha, bytes=output.stat().st_size,
                          command=command, build_seconds=time.monotonic() - before)
            _write(staging / 'geography.json', _json(geo))
            _write(staging / 'complete.json', _json(record))
            if hit:
                equal = rebuild_sha == saved['sha256']
                proof = dict(equal=equal, cached_sha256=saved['sha256'],
                             rebuilt_sha256=rebuild_sha, rebuild=str(staging),
                             build_seconds=record['build_seconds'])
                _write(staging / 'comparison.json', _json(proof))
                removed = dict(path=str(output), bytes=output.stat().st_size,
                               sha256=rebuild_sha, reason='rebuild comparison recorded')
                output.unlink()
                _write(staging / 'deleted.json', _json(removed))
                cache_result.setdefault('deleted', []).append(removed)
                if not equal:
                    raise RuntimeError(f"static rebuild is not byte-identical: {staging}")
                cache_result['rebuild_comparison'] = proof
            else:
                os.rename(staging, entry)
                saved = record
        cache_result.update(hit=hit, sha256=saved['sha256'], bytes=saved['bytes'],
                            original_build_seconds=saved['build_seconds'])
        out.parent.mkdir(parents=True, exist_ok=True)
        matches = out.exists() and digest(out, hashes) == saved['sha256']
        if out.exists() and not matches and not clobber:
            raise FileExistsError(f"existing static does not match this cache key: {out}; use a new output path")
        if not matches:
            # One physical prepared artifact per host. Consumers use a symlink;
            # the cache artifact is immutable and checked on every reuse.
            if clobber:
                temp = out.with_name(out.name + '.' + uuid.uuid4().hex + '.part')
                os.symlink(artifact.resolve(), temp)
                if out.exists() or out.is_symlink():
                    cache_result.setdefault('deleted', []).append(dict(
                        path=str(out), bytes=out.lstat().st_size, reason='explicit clobber'))
                os.replace(temp, out)
            else:
                os.symlink(artifact.resolve(), out)
        cache_result['artifact'] = str(artifact)
        cache_result['output_is_link'] = out.is_symlink()
        # Keep the engine receipt schema consumed by mesh_rows, with actual output path.
        builder = json.loads((entry / 'builder.json').read_bytes())
        builder['output_path'] = str(out)
        builder['static_cache'] = cache_result
        _write(receipt, _json(builder))
    cache_result['seconds'] = time.monotonic() - started
    _write(receipt.with_name(receipt.name + '.cache.json'), _json(cache_result))
    return cache_result


def warm_geography(geog, cache=None):
    """Fill the digest memo for one geography tree (an image bake runs this once), return its key."""
    root = Path(cache) if cache is not None else cache_dir()
    if root is None:
        raise ValueError('no static cache: set WOOF_HEX_STATIC_CACHE or WOOF_HEX_CACHE_ROOT, or pass --cache')
    started = time.monotonic()
    manifest = geography(geog, Path(root) / 'digests')
    return dict(geography_sha256=manifest['sha256'], files=len(manifest['files']),
                bytes=sum(item['bytes'] for item in manifest['files']),
                digests=str(Path(root) / 'digests'), seconds=round(time.monotonic() - started, 3))


if __name__ == '__main__':
    import argparse
    if sys.argv[1:2] == ['warm-geography']:
        warm = argparse.ArgumentParser(prog='python -m woof.hex.static_cache warm-geography',
                                       description=warm_geography.__doc__)
        warm.add_argument('geog', type=Path)
        warm.add_argument('--cache', type=Path)
        options = warm.parse_args(sys.argv[2:])
        print(json.dumps(warm_geography(options.geog, options.cache), indent=2))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'grid', 'geog', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--verify-rebuild', action='store_true')
    parser.add_argument('--nominal-dx-m')
    parser.add_argument('--valid-time')
    parser.add_argument('--terrain-only', action='store_true')
    args = vars(parser.parse_args())
    opts = []
    for name in ('nominal_dx_m', 'valid_time'):
        value = args.pop(name)
        if value is not None:
            opts += ['--' + name.replace('_', '-'), value]
    if args.pop('terrain_only'):
        opts += ['--terrain-only']
    print(json.dumps(build_static(**args, options=opts), indent=2))
