"""Exercise the slim install with local HTTP bytes and installed-only pins."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time
import urllib.request

import pytest
from woof.hex import assets
from woof.hex.cuda_backend import aot


@pytest.fixture
def estate(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    identity = {'nvrtc': [13, 4], 'nvrtc_library_sha256': 'f' * 64}
    name = 'kernel-pack-cuda13.json'
    data = json.dumps({'schema': 'gpuwm.kernel-pack/v1', 'compiler': identity, 'entries': []}).encode()
    (source / name).write_bytes(data)
    pins = {'schema': assets.SCHEMA, 'release_assets': {'repository': 'unused/test', 'tag': 'test'},
            'assets': {name: {'kind': 'kernel-pack', 'bytes': len(data),
                             'sha256': hashlib.sha256(data).hexdigest(),
                             'select': {'nvrtc_major': 13}, 'compiler': identity}}}
    manifest = tmp_path / 'pins.json'
    manifest.write_text(json.dumps(pins))
    monkeypatch.setattr(assets, 'PINS', manifest)
    monkeypatch.setattr(assets, '_VERIFIED', set())
    monkeypatch.setattr(assets, '_ATTEMPTS', {})
    monkeypatch.setattr(aot, '_WARNED', set())
    monkeypatch.setattr(aot, '_PACKS', {})
    monkeypatch.setattr(assets, 'checkout_asset', lambda name: None)
    monkeypatch.setenv('WOOF_HEX_PREP_CACHE', str(tmp_path / 'cache'))
    for key in (assets.FROM_ENV, assets.URL_ENV, assets.FIRST_USE_ENV, 'WOOF_HEX_AOT_PACK', 'WOOF_HEX_AOT'):
        monkeypatch.delenv(key, raising=False)
    return source, name, pins, identity


@pytest.fixture
def server(estate, monkeypatch):
    source, name, pins, identity = estate
    hits = []
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            super().do_GET()
        def log_message(self, *args):
            pass
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(source)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(assets.URL_ENV, f'http://127.0.0.1:{httpd.server_port}')
    yield hits
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('inspection/offline operation opened a socket')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def test_http_fetch_writes_verified_file_and_receipt(estate, server):
    source, name, pins, identity = estate
    result = assets.fetch_one(name, assets.load_pins())
    assert Path(result['path']).read_bytes() == (source / name).read_bytes()
    receipt = json.loads((assets.cache_root() / 'assets/receipts' / (name + '.json')).read_text())
    assert receipt['sha256'] == pins['assets'][name]['sha256']
    assert receipt['bytes'] == (source / name).stat().st_size
    assert receipt['url'].endswith('/' + name) and receipt['fetched_utc'] and receipt['wall_s'] >= 0
    assert server == ['/' + name]
    assert aot.Pack(identity=identity).reason is None
    assert server == ['/' + name]


def test_wrong_sha_rejected_without_cache_file(estate, server):
    source, name, pins, identity = estate
    pins['assets'][name]['sha256'] = '0' * 64
    with pytest.raises(assets.AssetError, match='sha256 mismatch'):
        assets.fetch_one(name, pins)
    assert not assets.asset_path(name, pins['assets'][name]).exists()
    assert not (assets.cache_root() / 'assets/receipts' / (name + '.json')).exists()
    assert not list(assets.cache_root().rglob('.partial-*'))


def test_unreachable_once_falls_back_to_compile(estate, monkeypatch, capsys):
    source, name, pins, identity = estate
    with socket.socket() as closed:
        closed.bind(('127.0.0.1', 0))
        port = closed.getsockname()[1]
    monkeypatch.setenv(assets.URL_ENV, f'http://127.0.0.1:{port}')
    attempts = []
    original = urllib.request.urlopen
    def opening(*args, **kwargs):
        attempts.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(urllib.request, 'urlopen', opening)
    with ThreadPoolExecutor(max_workers=4) as pool:
        packs = list(pool.map(lambda _: aot.Pack(identity=identity), range(4)))
    assert all(p.select('missing', 'source', (), 'sm_120') == (None, 'pack_missing') for p in packs)
    assert len(attempts) == 1 and attempts[0][1]['timeout'] == 20
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and 'offline' in lines[0] and 'NVRTC' in lines[0]
    assert 'woof hex fetch-assets' in lines[0] and 'WOOF_HEX_ASSETS_FROM' in lines[0]
    assert 'Connection refused' in next(iter(assets._ATTEMPTS.values()))


def test_offline_directory_has_no_network(estate, monkeypatch):
    source, name, pins, identity = estate
    no_network(monkeypatch)
    monkeypatch.setenv(assets.FROM_ENV, str(source))
    path = aot.default_pack(identity)
    assert path == assets.asset_path(name, pins['assets'][name])
    receipt = json.loads((assets.cache_root() / 'assets/receipts' / (name + '.json')).read_text())
    assert receipt['source_dir'] == str(source) and 'url' not in receipt


def test_first_use_off_never_fetches(estate, monkeypatch, capsys):
    no_network(monkeypatch)
    monkeypatch.setenv(assets.FIRST_USE_ENV, 'off')
    assert aot.default_pack(estate[3]) is None
    assert aot.default_pack(estate[3]) is None
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and 'first-use off' in lines[0]
    assert not assets.cache_root().exists()


def test_doctor_never_fetches_and_reports_corruption(estate, monkeypatch):
    from woof.hex import doctor
    no_network(monkeypatch)
    source, name, pins, identity = estate
    rows = doctor.check_on_demand_assets()
    assert rows[0].status == doctor.INFO
    assert 'on demand' in rows[0].detail and 'first use' in rows[0].detail
    assert rows[0].remedy == f'woof hex fetch-assets --only {name}'
    assert not assets.cache_root().exists()
    assets.fetch_one(name, pins, from_directory=source)
    assert doctor.check_on_demand_assets()[0].status == doctor.VERIFIED
    path = assets.asset_path(name, pins['assets'][name])
    path.write_bytes(b'x' * path.stat().st_size)
    row = doctor.check_on_demand_assets()[0]
    assert row.status == doctor.MISSING and row.required
    assert 'sha256 mismatch' in row.detail and 'woof hex fetch-assets' in row.detail


def test_failed_batch_leaves_no_cache_files(estate, capsys):
    source, name, pins, identity = estate
    other = 'second-pack.json'
    (source / other).write_bytes((source / name).read_bytes())
    pins['assets'][other] = dict(pins['assets'][name])
    assets.PINS.write_text(json.dumps(pins))
    original = (source / name).read_bytes()
    (source / name).write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert assets.main(['fetch-assets', '--from', str(source)]) == 1
    assert name in capsys.readouterr().err
    assert not assets.cache_root().exists()


def test_doctor_sees_offline_bytes_without_staging(estate, monkeypatch):
    from woof.hex import doctor
    source, name, pins, identity = estate
    monkeypatch.setenv(assets.FROM_ENV, str(source))
    no_network(monkeypatch)
    finding = doctor.check_on_demand_assets()[0]
    assert finding.status == doctor.VERIFIED
    assert finding.evidence['path'] == str(source / name)
    assert not assets.cache_root().exists()


def test_doctor_scope_guards_incidental_resolution(estate, monkeypatch):
    from woof.hex import doctor
    no_network(monkeypatch)
    monkeypatch.setattr(doctor, 'collect', lambda: (aot.default_pack(estate[3]), [])[1])
    assert doctor.run_doctor(argparse.Namespace(json=True)) == 0
    assert not assets.cache_root().exists()
    assert assets._ATTEMPTS == {}


def test_engine_inspection_scope_never_fetches(estate, monkeypatch):
    from woof.bridges import inspection_only
    no_network(monkeypatch)
    with inspection_only():
        assert aot.default_pack(estate[3]) is None
    assert assets._ATTEMPTS == {}


def test_list_no_network(estate, monkeypatch, capsys):
    no_network(monkeypatch)
    assert assets.main(['fetch-assets', '--list', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['assets'][0]['status'] == 'on demand'
    assert not assets.cache_root().exists()


def test_compiler_mismatch_and_other_family_never_download(estate, monkeypatch, capsys):
    no_network(monkeypatch)
    identity = {**estate[3], 'nvrtc_library_sha256': '0' * 64}
    assert aot.default_pack(identity) is None
    assert aot.default_pack(identity) is None
    assert len(capsys.readouterr().err.splitlines()) == 1
    assert aot.default_pack({'nvrtc': [12, 9]}) is None
    assert assets._ATTEMPTS == {}


def test_default_fetch_is_parallel_and_only_filters(estate, monkeypatch, capsys):
    source, name, pins, identity = estate
    pins['assets']['second.json'] = {**pins['assets'][name]}
    (source / 'second.json').write_bytes((source / name).read_bytes())
    assets.PINS.write_text(json.dumps(pins))
    barrier = threading.Barrier(2, timeout=5)
    original = assets.fetch_one
    def concurrent(*args, **kwargs):
        barrier.wait()
        return original(*args, **kwargs)
    monkeypatch.setattr(assets, 'fetch_one', concurrent)
    assert assets.main(['fetch-assets', '--from', str(source), '--json']) == 0
    assert len(json.loads(capsys.readouterr().out)['assets']) == 2
    monkeypatch.setattr(assets, 'fetch_one', original)
    assert assets.main(['fetch-assets', '--list', '--only', name]) == 0
    assert 'second.json' not in capsys.readouterr().out
    assert assets.main(['fetch-assets', '--only', 'unknown']) == 1
    assert 'unknown' in capsys.readouterr().err


def test_bad_offline_bytes_fail_cli(estate, capsys):
    source, name, pins, identity = estate
    (source / name).write_bytes(b'wrong')
    assert assets.main(['fetch-assets', '--from', str(source)]) == 1
    error = capsys.readouterr().err
    assert name in error and 'bytes mismatch' in error


def test_cache_off_still_has_per_user_asset_root(estate, monkeypatch, tmp_path):
    monkeypatch.setenv('WOOF_HEX_PREP_CACHE', 'off')
    monkeypatch.delenv('WOOF_HEX_CACHE_ROOT', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    assert assets.cache_root() == tmp_path / 'xdg/woof hex'


def test_explicit_override_precedes_pins(estate, monkeypatch):
    path = estate[0] / estate[1]
    monkeypatch.setenv('WOOF_HEX_AOT_PACK', str(path))
    monkeypatch.setattr(assets, 'load_pins', lambda: pytest.fail('explicit override read pins'))
    assert aot.default_pack(estate[3]) == path
    assert aot.Pack(identity=estate[3]).reason is None


def test_private_checkout_only_accepts_pin(estate, monkeypatch, capsys):
    source, name, pins, identity = estate
    path = source / name
    monkeypatch.setattr(assets, 'checkout_asset', lambda name: path)
    no_network(monkeypatch)
    assert aot.default_pack(identity) == path
    path.write_bytes(b'x' * path.stat().st_size)
    monkeypatch.setenv(assets.FIRST_USE_ENV, 'off')
    assert aot.default_pack(identity) is None
    assert 'sha256 mismatch' in capsys.readouterr().err


def test_select_downloads_only_matching_family(estate, server):
    source, name, pins, identity = estate
    pins['assets']['kernel-pack-cuda12.json'] = {
        **pins['assets'][name], 'select': {'nvrtc_major': 12},
        'compiler': {**identity, 'nvrtc': [12, 9]}}
    assets.PINS.write_text(json.dumps(pins))
    assert aot.default_pack(identity).name == name
    assert server == ['/' + name]


def test_staged_resolution_memoises_hash(estate, monkeypatch):
    source, name, pins, identity = estate
    assets.fetch_one(name, pins, from_directory=source)
    assets._VERIFIED.clear()
    original = hashlib.file_digest
    calls = []
    def hashing(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(hashlib, 'file_digest', hashing)
    assert aot.default_pack(identity) == aot.default_pack(identity)
    assert len(calls) == 1


def test_pin_subcommand_is_deterministic(estate, tmp_path):
    source, name, pins, identity = estate
    compiler = {**identity, 'cupy': 'test', 'cupy_package': 'test', 'nvrtc_package': 'test',
                'cache_key': 'test', 'use_ptx': False, 'use_pch': False, 'debug': '0'}
    for major in (12, 13):
        (source / f'kernel-pack-cuda{major}.json').write_text(
            json.dumps({'compiler': {**compiler, 'nvrtc': [major, 4]}}))
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    for output in (first, second):
        assert assets.main(['pin', '--from', str(source), '--output', str(output)]) == 0
    assert first.read_bytes() == second.read_bytes()
    for name, row in json.loads(first.read_text())['assets'].items():
        assert row['compiler'] == json.loads((source / name).read_text())['compiler']
        assert row['sha256'] == hashlib.sha256((source / name).read_bytes()).hexdigest()


def test_pack_matches_the_loaded_compiler_not_pip_metadata():
    """CuPy loaded the system CUDA 13.0 libnvrtc while pip metadata named 13.4.92."""
    from woof.hex.kernel_pack import same_compiler
    built = {"cache_key": "k", "cupy": "14.2.0", "cupy_package": "cupy-cuda13x==14.2.0", "debug": "0",
             "nvrtc": [13, 0], "nvrtc_library_sha256": "a" * 64, "nvrtc_package": "nvidia-cuda-nvrtc==13.0.88",
             "use_pch": False, "use_ptx": False}
    measured = {**built, "nvrtc_package": "nvidia-cuda-nvrtc==13.4.92"}
    assert same_compiler(built, measured)
    assert not same_compiler(built, {**measured, "nvrtc_library_sha256": "b" * 64})
    assert not same_compiler(built, {**measured, "nvrtc": [13, 4]})


def test_a_pin_row_may_name_its_own_release():
    pins = {"release_assets": {"repository": "o/r", "tag": "kernel-packs-v0.4.2"},
            "assets": {"kernel-pack-cuda13.0.json": {"release_tag": "kernel-packs-v0.4.5"},
                       "kernel-pack-cuda13.json": {}}}
    assert assets.asset_url("kernel-pack-cuda13.0.json", pins).endswith("/kernel-packs-v0.4.5/kernel-pack-cuda13.0.json")
    assert assets.asset_url("kernel-pack-cuda13.json", pins).endswith("/kernel-packs-v0.4.2/kernel-pack-cuda13.json")


def test_a_pack_the_box_image_holds_is_used_by_a_run_with_its_own_cache(estate, monkeypatch, tmp_path):
    """THE BREAKAGE (an RTX 5090 host and V8X6000, 2026-10-29): packs staged on the box were invisible to a run with its
    own cache root; the run fetched an unpublished tag, got a 404 and compiled every kernel through NVRTC."""
    source, name, pins, identity = estate
    no_network(monkeypatch)
    monkeypatch.setenv(assets.FIRST_USE_ENV, 'off')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    image = tmp_path / 'image'
    row = pins['assets'][name]
    (image / row['sha256']).mkdir(parents=True)
    (image / row['sha256'] / name).write_bytes(b'x' * row['bytes'])  # right size, wrong bytes: never used
    monkeypatch.setenv(assets.SYSTEM_ENV, str(image))
    assert aot.default_pack(identity) is None
    (image / row['sha256'] / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr(assets, '_ATTEMPTS', {})
    assert aot.default_pack(identity) == image / row['sha256'] / name
    assert not assets.cache_root().exists()
    # The box's default cache (XDG) serves it too, in its <sha256>/<name> layout.
    monkeypatch.setenv(assets.SYSTEM_ENV, '')
    default = tmp_path / 'xdg' / 'woof hex' / 'prep' / 'assets' / row['sha256']
    default.mkdir(parents=True)
    (default / name).write_bytes((source / name).read_bytes())
    assert assets.resolve_pack(identity) == (default / name, None)
