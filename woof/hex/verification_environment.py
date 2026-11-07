"""Strict landing environment, captured by the forecast's Python installation.

verify-mpas captures the environment itself, before and after the forecast it
runs, and every receipt carries both captures: the receipt is the record, so a
deck needs no hand-captured one.  What is enforced, each with the breakage it
prevents:

* numerics=strict.  Fast numerics contracts multiply-adds and rounds division
  and square root differently, so its words are not native MPAS's; the bitwise
  claim is "with strict numerics".
* an authenticated engine identity: a clean source checkout's git SHA, the
  archive SHA-256 pip records for a wheel-file install, or (a PyPI install,
  where pip records neither) the installed files verified against their RECORD
  hashes, folded into ``engine_content_sha256``.  That digest is a function of
  the wheel's own RECORD rows, so anyone can recompute it from the published
  wheel (``wheel_content_sha256``).  Without it a receipt could name an engine
  whose bytes were edited after install.
* a complete capture (CUDA runtime, NVRTC, CuPy, driver, device, every
  external math header): a verdict must state the environment it holds in.
* the same environment after the forecast as before it.

A deck's ``strict_environment`` is optional.  When present it is the
reference a regression is attributed against: a capture that differs from it
cannot turn word differences into a source regression or advance the ratchet
(verdict ENVIRONMENT, raw counts kept).  Words identical to native are the
proof itself and stay GREEN in any environment; the receipt lists how the
as-run environment differs from the reference.

Comparison-only output must still carry its as-run before/after capture;
inspecting an installation after the fact cannot authenticate an old forecast.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


SCHEMA = 'mpas-strict-environment-v1'
KEYS = ('engine_version', 'engine_git_sha', 'engine_record_sha256', 'engine_wheel_sha256',
        'engine_content_sha256', 'math_headers', 'cuda_toolkit', 'cuda_runtime', 'nvrtc',
        'cupy_version', 'driver', 'gpu_uuid', 'numerics')
IDENTITY_KEYS = ('engine_git_sha', 'engine_record_sha256', 'engine_wheel_sha256', 'engine_content_sha256')
# pip rewrites RECORD at install (console-script shebangs carry the venv's
# absolute python path), so its text names a venv, not an engine: recorded as
# provenance, never compared between two installations.
REFERENCE_KEYS = tuple(key for key in KEYS if key != 'engine_record_sha256')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_rows(rows):
    """``(path, sha256 hex)`` for the files a wheel installs into site-packages.

    Skips rows without a hash (RECORD itself, compiled .pyc), the dist-info
    (INSTALLER and direct_url.json differ by installer) and anything outside
    site-packages (console scripts, whose shebangs name the venv; a wheel's
    .data tree, which installs outside it).
    """
    kept = []
    for path, digest in rows:
        parts = PurePosixPath(path).parts
        if not digest or '..' in parts or parts[0].endswith(('.dist-info', '.data')):
            continue
        algorithm, _, value = digest.partition('=')
        if algorithm != 'sha256':
            raise ValueError(f'RECORD row {path} is not sha256')
        kept.append((path, base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)).hex()))
    return sorted(kept)


def _fold(rows):
    payload = hashlib.sha256()
    for path, digest in rows:
        payload.update(path.encode() + b'\0' + bytes.fromhex(digest))
    return payload.hexdigest()


def wheel_content_sha256(wheel):
    """``engine_content_sha256`` of a wheel file, from its own RECORD (no install)."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist()
                    if n.endswith('.dist-info/RECORD') and n.count('/') == 1)
        text = archive.read(name).decode()
    rows = [line.rsplit(',', 2)[:2] for line in text.splitlines() if line.strip()]
    return _fold(_content_rows(rows))


def installed_content(dist):
    """Verify every installed file against its RECORD hash; return (digest, errors)."""
    files = [f for f in (dist.files or ()) if f.hash is not None]
    rows = _content_rows([(str(f), f'{f.hash.mode}={f.hash.value}') for f in files])
    located = {str(f): Path(dist.locate_file(f)) for f in files}

    def check(row):
        path, expected = row
        try:
            actual = sha(located[path])
        except OSError as error:
            return f'installed engine file unreadable: {path}: {error}'
        return None if actual == expected else f'installed engine file differs from its RECORD: {path}'

    # Hashing releases the GIL: every core, not one.
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        errors = [e for e in pool.map(check, rows) if e]
    if not rows:
        errors.append('installed engine RECORD lists no hashed files')
    return _fold(rows), errors


def capture(numerics, command=()):
    """Read the same engine override/import search path as the model command.

    Pin every external .cuh, a conservative superset of the headers inlined
    by the hex kernels. Paths are provenance, relative names and bytes gate.
    No CUDA kernel is compiled or launched here.
    """
    record = dict(schema=SCHEMA, pins={'numerics': numerics}, errors=[])
    pins, errors = record['pins'], record['errors']
    try:
        from woof.hex import engine_pin
        dist = importlib.metadata.distribution(engine_pin.DISTRIBUTION)
        pins['engine_version'] = dist.version
        installed_record = dist.read_text('RECORD')
        pins['engine_record_sha256'] = (hashlib.sha256(installed_record.encode()).hexdigest()
                                         if installed_record else None)
        spec = importlib.util.find_spec('woof')
        root = Path(next(iter(spec.submodule_search_locations))).parent
        overrides = [i for i, value in enumerate(command) if value == '--gpuwm-checkout'
                     or value.startswith('--gpuwm-checkout=')]
        if len(overrides) > 1:
            raise ValueError('duplicate --gpuwm-checkout')
        if overrides:
            i = overrides[0]
            root = Path(command[i].split('=', 1)[1] if '=' in command[i] else command[i+1]).resolve()
            pins['engine_version'] = engine_pin.checkout_version(root)
            # The source override is the engine in use; pip RECORD describes
            # a different installation and must not impersonate its bytes.
            pins['engine_record_sha256'] = None
        kernels = root / 'woof/core/kernels'
        pins['math_headers'] = {str(p.relative_to(kernels)).replace(os.sep, '/'): sha(p)
                                for p in sorted(kernels.rglob('*.cuh'))}
        record['engine_root'] = str(root.resolve())
        pins['engine_git_sha'] = None
        pins['engine_wheel_sha256'] = None
        pins['engine_content_sha256'] = None
        # A wheel's source revision is accepted only from installed metadata,
        # never from an unrelated enclosing checkout or an environment string.
        direct = dist.read_text('direct_url.json')
        if direct and not overrides:
            metadata = json.loads(direct)
            pins['engine_git_sha'] = metadata.get('vcs_info', {}).get('commit_id')
            pins['engine_wheel_sha256'] = metadata.get('archive_info', {}).get('hashes', {}).get('sha256')
        locate = getattr(dist, 'locate_file', None)
        installed_root = Path(locate('')).resolve() if locate else None
        if (not overrides and installed_record and installed_root is not None
                and root.resolve().is_relative_to(installed_root)):
            # The imported engine is this distribution's installed files: a
            # PyPI install has no archive hash, so its identity is its files.
            pins['engine_content_sha256'], content_errors = installed_content(dist)
            errors.extend(content_errors)
        if (root / '.git').exists():
            git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                 capture_output=True, text=True, check=True)
            pins['engine_git_sha'] = git.stdout.strip()
            dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                                    '--untracked-files=no'], capture_output=True, text=True, check=True)
            if dirty.stdout.strip():
                errors.append('engine checkout has tracked modifications')
    except Exception as error:
        errors.append('engine capture: ' + str(error))
    try:
        import cupy as cp
        from cupy.cuda import nvrtc, runtime
        pins['cupy_version'] = cp.__version__
        pins['cuda_runtime'] = runtime.runtimeGetVersion()
        pins['nvrtc'] = list(nvrtc.getVersion())
        device = runtime.getDevice()
        bus = cp.cuda.Device(device).pci_bus_id
        info = subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), '--id=' + bus,
                               '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        uuid, driver = info.stdout.strip().split(',')
        pins['gpu_uuid'], pins['driver'] = uuid.strip(), driver.strip()
        try:
            nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            pins['cuda_toolkit'] = nvcc.stdout.strip()
        except FileNotFoundError:
            # CUDA Python wheels ship an NVRTC toolkit without the nvcc CLI.
            # Pin the installed toolkit and compiler/runtime package versions;
            # the loaded NVRTC and runtime API versions are pinned above too.
            pins['cuda_toolkit'] = {name: importlib.metadata.version(name) for name in
                ('cuda-toolkit', 'nvidia-cuda-nvrtc', 'nvidia-cuda-runtime')}
    except Exception as error:
        errors.append('CUDA capture: ' + str(error))
    return record


def problems(record):
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        return ['missing as-run strict environment']
    pins = record.get('pins', {})
    errors = list(record.get('errors', []))
    for key in KEYS:
        if not pins.get(key) and key not in IDENTITY_KEYS:
            errors.append('missing environment pin: ' + key)
    if not re.fullmatch(r'[0-9a-f]{40}', str(pins.get('engine_git_sha', ''))):
        # An installed archive has no checkout commit: its identity is the
        # archive bytes pip recorded plus RECORD, or (a PyPI install, where pip
        # records no archive hash) the installed files verified against RECORD.
        archive = (re.fullmatch(r'[0-9a-f]{64}', str(pins.get('engine_wheel_sha256', '')))
                   and re.fullmatch(r'[0-9a-f]{64}', str(pins.get('engine_record_sha256', ''))))
        content = re.fullmatch(r'[0-9a-f]{64}', str(pins.get('engine_content_sha256', '')))
        if not (archive or content):
            errors.append('engine identity needs a clean source git SHA, an installed wheel SHA256 '
                          'with its RECORD, or installed files verified against RECORD')
    headers = pins.get('math_headers', {})
    if not isinstance(headers, dict) or not headers or any(
            not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in headers.values()):
        errors.append('external math header hashes are incomplete')
    if pins.get('numerics') != 'strict':
        errors.append('strict landing requires numerics=strict')
    return errors


def judge(expected, before, after):
    """ENVIRONMENT: an incomplete or changing as-run capture (no verdict is stated).

    PINNED / DIFFERS: the deck carries a reference and the capture equals it /
    differs from it (``reference_differences``). RECORDED: the deck carries no
    reference; the receipt's own capture is the record.
    """
    failures = []
    for label, record in (('before', before), ('after', after)):
        failures.extend(label + ': ' + error for error in problems(record))
    if not failures:
        failures.extend('environment changed during the run: ' + key for key in KEYS
                        if before['pins'].get(key) != after['pins'].get(key))
    differences = []
    if expected is not None:
        reference = problems(expected)
        failures.extend('reference: ' + error for error in reference)
        if not reference and not failures:
            # A legacy reference names fewer pins; only what it names is compared.
            differences = [key for key in REFERENCE_KEYS if key in expected['pins']
                           and expected['pins'][key] != before['pins'].get(key)]
    verdict = ('ENVIRONMENT' if failures else 'RECORDED' if expected is None else
               'DIFFERS' if differences else 'PINNED')
    return dict(verdict=verdict, failures=failures, reference_differences=differences,
                reference=expected, before=before, after=after)
