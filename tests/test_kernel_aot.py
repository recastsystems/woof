"""A corrupt or stale image must never become a cached kernel."""
import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from woof.kernel_aot import Pack, source_hash


def row(arch="sm_120"):
    blob = b"\x7fELFtest-image"
    return {"module_key": "hexcore.test", "source": "source", "arch": arch,
            "source_sha256": source_hash("source"), "options": ["--fmad=false"],
            "cubin_base64": base64.b64encode(blob).decode(),
            "cubin_sha256": hashlib.sha256(blob).hexdigest()}


def test_shipped_conditional_headers_bind_the_same_complete_trees_as_retained_images(tmp_path, monkeypatch):
    from woof.kernel_dependencies import include_tree_dependencies
    root = tmp_path / 'headers'
    root.mkdir()
    header = root / 'branch.h'
    header.write_text('#define VALUE 1\n')
    source = '#if ENABLED\n#include <branch.h>\n#endif\n'
    options = ['-I' + str(root)]
    bound = dict(row(), source=source, source_sha256=source_hash(source), options=options,
                 include_trees=include_tree_dependencies(options, {}))
    selected = pack(tmp_path, monkeypatch, [bound])
    assert selected.select('hexcore.test', source, options, 'sm_120')[1] is None
    # A new process/session must reject changed AND newly shadowing headers.
    header.write_text('#define VALUE 2\n')
    selected_path = tmp_path / 'pack.json'
    assert Pack(selected_path, identity={'compiler': 'test'}).select(
        'hexcore.test', source, options, 'sm_120')[1] == 'header_tree_mismatch'
    header.write_text('#define VALUE 1\n')
    (root/'new.h').write_text('#define ADDED 1\n')
    assert Pack(selected_path, identity={'compiler': 'test'}).select(
        'hexcore.test', source, options, 'sm_120')[1] == 'header_tree_mismatch'


def pack(tmp_path, monkeypatch, rows=None):
    write = pytest.importorskip("woof.kernel_pack").write
    monkeypatch.delenv("WOOF_AOT", raising=False)
    path = tmp_path / "pack.json"
    write(path, {"compiler": "test"}, rows or [row()])
    return Pack(path, identity={"compiler": "test"})


@pytest.mark.parametrize("key,source,options,sm,reason", [
    ("hexcore.test", "source", ["--fmad=false"], "sm_89", "architecture_mismatch"),
    ("hexcore.test", "source", ["--fmad=false"], "sm_100", "architecture_mismatch"),
    ("hexcore.test", "changed", ["--fmad=false"], "sm_120", "source_hash_mismatch"),
    ("hexcore.test", "source", [], "sm_120", "options_mismatch"),
    ("hexcore.other", "source", ["--fmad=false"], "sm_120", "module_missing"),
])
def test_named_fallbacks(tmp_path, monkeypatch, key, source, options, sm, reason):
    assert pack(tmp_path, monkeypatch).select(key, source, options, sm) == (None, reason)


def test_rawmodule_load_and_corrupt_local_file_recovery(tmp_path, monkeypatch):
    selected = pack(tmp_path, monkeypatch)
    calls = []
    def raw_module(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(module="loaded")
    cp = SimpleNamespace(RawModule=raw_module)
    for _ in range(2):
        module, reason, entry = selected.load(cp, tmp_path, "hexcore.test", "source", ["--fmad=false"], "sm_120")
        assert module.module == "loaded" and reason is None
        path = tmp_path / "gpuwm-aot" / (entry["cubin_sha256"] + ".cubin")
        assert path.read_bytes() == b"\x7fELFtest-image"
        path.write_bytes(b"corrupt")
    assert all(set(call) == {"path"} for call in calls)


def test_corrupt_digest_never_loads(tmp_path, monkeypatch):
    broken = row()
    broken["cubin_sha256"] = "0" * 64
    cp = SimpleNamespace(RawModule=lambda **kwargs: pytest.fail("corrupt image loaded"))
    assert pack(tmp_path, monkeypatch, [broken]).load(cp, tmp_path, "hexcore.test", "source", ["--fmad=false"], "sm_120") == (None, "cubin_digest_mismatch", None)


def test_envelope_corruption_and_compiler_change(tmp_path, monkeypatch):
    selected = pack(tmp_path, monkeypatch)
    path = tmp_path / "pack.json"
    assert Pack(path, identity={"compiler": "changed"}).reason == "pack_invalid_or_compiler_mismatch"
    original_stat = path.stat()
    document = json.loads(path.read_text())
    document["entries"][0]["arguments"]["source"] = "altered"
    path.write_text(json.dumps(document))
    assert path.stat().st_size == original_stat.st_size
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert Pack(path, identity={"compiler": "test"}).reason == "pack_invalid_or_compiler_mismatch"


@pytest.mark.parametrize("major,explicit", [(12, None), (13, None), (12, "argument"), (12, "environment")])
def test_default_compiler_families_and_explicit_pack(tmp_path, monkeypatch, major, explicit):
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    primary = tmp_path / "kernel-pack.json"
    sidecar = tmp_path / "kernel-pack-cuda12.json"
    identities = {n: {"nvrtc": [n, 9]} for n in (12, 13)}
    write(primary, identities[13], [row()])
    write(sidecar, identities[12], [row()])
    monkeypatch.setattr(aot, "DEFAULT_PACK", primary)
    monkeypatch.delenv("WOOF_AOT", raising=False)
    monkeypatch.delenv("WOOF_AOT_PACK", raising=False)
    if explicit == "environment":
        monkeypatch.setenv("WOOF_AOT_PACK", str(primary))
    selected = Pack(primary if explicit == "argument" else None, identity=identities[major])
    if explicit:
        assert selected.reason == "pack_invalid_or_compiler_mismatch"
    else:
        assert selected.reason is None
        assert selected.compiler == identities[major]
    # A filename cannot admit a different minor compiler or corrupt envelope.
    if major == 12 and not explicit:
        assert Pack(identity={"nvrtc": [12, 8]}).reason == "pack_invalid_or_compiler_mismatch"
        primary.unlink()
        assert Pack(identity=identities[12]).reason is None
        sidecar.write_text('{"schema": "corrupt"}')
        assert Pack(identity=identities[12]).reason == "pack_invalid_or_compiler_mismatch"


@pytest.mark.parametrize('source,reason', [
    ('#include "owned.h"\n', 'external_header_dependency'),
    ('#define H "owned.h"\n#include H\n', 'unbound_header_dependency'),
])
def test_shipped_pack_without_header_binding_declines_request(tmp_path, monkeypatch, source, reason):
    (tmp_path / 'owned.h').write_text('#define VALUE 1\n')
    entry = row();entry.update(source=source, source_sha256=source_hash(source), options=['-I'+str(tmp_path)])
    selected = pack(tmp_path, monkeypatch, [entry])
    assert selected.select('hexcore.test', source, entry['options'], 'sm_120') == (None, reason)


@pytest.mark.parametrize('primary', ['matching', 'older', 'missing', 'corrupt', 'corrupt-entry'])
@pytest.mark.parametrize('extension_name', ['gfdleta', 'mskf'])
def test_extension_on_cold_compiler_family(tmp_path, monkeypatch, primary, extension_name):
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'exact-bytes'}
    monkeypatch.setattr(aot, 'DEFAULT_PACK', tmp_path/'kernel-pack.json')
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    if primary != 'missing':
        write(aot.DEFAULT_PACK, identity if primary != 'older' else {'nvrtc': [13, 2]}, [row()])
        if primary == 'corrupt':
            aot.DEFAULT_PACK.write_text('not a kernel pack')
        elif primary == 'corrupt-entry':
            document = json.loads(aot.DEFAULT_PACK.read_text())
            document['entries'][0]['sha256'] = '0' * 64
            aot.DEFAULT_PACK.write_text(json.dumps(document))
    extra = dict(row(), module_key='nvrtc.direct.' + extension_name)
    path = getattr(aot, extension_name + '_pack_path')(identity)
    write(path, identity, [extra])
    held = Pack(identity=identity)
    assert held.reason is None
    assert held.select(extra['module_key'], 'source', ['--fmad=false'], 'sm_120')[1] is None
    assert len(held.rows) == (2 if primary == 'matching' else 1)
    # A different compiler patch, stale source, or damaged image cannot use it.
    assert Pack(identity={**identity, 'build': 'changed'}).reason is not None
    assert held.select(extra['module_key'], 'changed', ['--fmad=false'], 'sm_120')[1] == 'source_hash_mismatch'
    damaged = dict(extra, cubin_sha256='0'*64)
    write(path, identity, [damaged])
    held = Pack(identity=identity)
    selected, reason = held.select(extra['module_key'], 'source', ['--fmad=false'], 'sm_120')
    assert reason is None and held.image(selected) is None
    # Explicit pack selection remains authoritative; no implicit extension.
    explicit = tmp_path/'explicit.json'
    write(explicit, identity, [row()])
    assert Pack(explicit, identity=identity).select(extra['module_key'], 'source', ['--fmad=false'], 'sm_120')[1] == 'module_missing'


@pytest.mark.parametrize('primary', [False, True])
def test_mskf_and_gfdl_extensions_keep_both_indexes(tmp_path, monkeypatch, primary):
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'bound-bytes'}
    monkeypatch.setattr(aot, 'DEFAULT_PACK', tmp_path / 'kernel-pack.json')
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    entries = [dict(row(), module_key='nvrtc.direct.' + name) for name in ('gfdleta', 'mskf')]
    for name, entry in zip(('gfdleta', 'mskf'), entries):
        write(getattr(aot, name + '_pack_path')(identity), identity, [entry])
    if primary:
        write(aot.DEFAULT_PACK, identity, [row()])
    held = Pack(identity=identity)
    assert held.reason is None and len(held.rows) == 2 + int(primary)
    for entry in entries:
        selected, reason = held.select(entry['module_key'], 'source', ['--fmad=false'], 'sm_120')
        assert reason is None and held.image(selected) is not None


def test_startup_companion_loads_exact_compiler_images(tmp_path, monkeypatch):
    import sys
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'startup-companion'}
    primary = tmp_path / 'family' / 'kernel-pack.json'
    startup = tmp_path / 'startup'
    monkeypatch.setattr(aot, 'DEFAULT_PACK', primary)
    monkeypatch.setattr(aot, '_default_pack_path', lambda: primary)
    monkeypatch.setitem(sys.modules, 'gpuwm_kernels_startup',
                        SimpleNamespace(data_path=lambda: startup))
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    extra = dict(row(), module_key='cupy.cache.startup')
    path = aot.compiler_pack_path(identity)
    assert path.parent == startup / 'kernel-packs'
    write(path, identity, [extra])
    held = Pack(identity=identity)
    selected, reason = held.select(extra['module_key'], 'source', ['--fmad=false'], 'sm_120')
    assert reason is None and held.image(selected) == b'\x7fELFtest-image'
    assert Pack(identity={**identity, 'build': 'other'}).reason is not None
    assert held.select(extra['module_key'], 'changed', ['--fmad=false'], 'sm_120')[1] == 'source_hash_mismatch'


def test_custom_pack_never_inherits_installed_startup_images(tmp_path, monkeypatch):
    import sys
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'startup-companion'}
    default = tmp_path / 'family' / 'kernel-pack.json'
    custom = tmp_path / 'custom' / 'kernel-pack.json'
    startup = tmp_path / 'startup'
    monkeypatch.setattr(aot, 'DEFAULT_PACK', default)
    monkeypatch.setattr(aot, '_default_pack_path', lambda: default)
    monkeypatch.setitem(sys.modules, 'gpuwm_kernels_startup',
                        SimpleNamespace(data_path=lambda: startup))
    write(aot.compiler_pack_path(identity), identity, [row()])
    monkeypatch.setattr(aot, 'DEFAULT_PACK', custom)
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    assert aot.compiler_pack_path(identity).parent == custom.parent / 'kernel-packs'
    assert Pack(identity=identity).reason is not None


def test_exact_compiler_patch_pack_precedes_family_and_keeps_gfdl(tmp_path, monkeypatch):
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'exact-compiler-bytes'}
    monkeypatch.setattr(aot, 'DEFAULT_PACK', tmp_path / 'kernel-pack.json')
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    write(tmp_path / 'kernel-pack-cuda13.json', {'nvrtc': [13, 2]}, [row()])
    general = dict(row(), module_key='cupy.cache.test')
    gfdl = dict(row(), module_key='nvrtc.direct.gfdleta')
    profile = aot.compiler_pack_path(identity)
    write(profile, identity, [general])
    write(aot.gfdleta_pack_path(identity), identity, [gfdl])
    held = Pack(identity=identity)
    assert held.reason is None and len(held.rows) == 2
    for entry in (general, gfdl):
        assert held.select(entry['module_key'], 'source', ['--fmad=false'], 'sm_120')[1] is None
    assert Pack(identity={**identity, 'build': 'other-bytes'}).reason is not None
    # Explicit selection and AOT-off never select a hidden profile.
    explicit = tmp_path / 'explicit.json'
    write(explicit, identity, [general])
    assert len(Pack(explicit, identity=identity).rows) == 1
    monkeypatch.setenv('WOOF_AOT', 'off')
    assert Pack(identity=identity).reason == 'aot_disabled'


@pytest.mark.parametrize('primary', ['matching', 'older'])
def test_unit_pack_families_merge_mskf_beside_gfdl(tmp_path, monkeypatch, primary):
    """cu_physics=11 compiled its 255-register MSKF column on one core inside
    the first forecast step (306 s, SKEPTIC-3 batch 2 item 2) because GFDL was
    the only compiler-bound unit supplement. Every family rides the same pack."""
    from woof import kernel_aot as aot
    from woof.kernel_pack import write
    identity = {'nvrtc': [13, 4], 'build': 'exact-bytes'}
    monkeypatch.setattr(aot, 'DEFAULT_PACK', tmp_path/'kernel-pack.json')
    monkeypatch.delenv('WOOF_AOT', raising=False)
    monkeypatch.delenv('WOOF_AOT_PACK', raising=False)
    write(aot.DEFAULT_PACK, identity if primary == 'matching' else {'nvrtc': [13, 2]}, [row()])
    gfdl = dict(row(), module_key='nvrtc.direct.gfdleta')
    mskf = dict(row(), module_key='nvrtc.direct.mskf')
    write(aot.unit_pack_path('gfdleta', identity), identity, [gfdl])
    write(aot.unit_pack_path('mskf', identity), identity, [mskf])
    held = Pack(identity=identity)
    assert held.reason is None
    assert len(held.rows) == (3 if primary == 'matching' else 2)
    for entry in (gfdl, mskf):
        assert held.select(entry['module_key'], 'source', ['--fmad=false'], 'sm_120')[1] is None
    # Another compiler's MSKF images are never substituted.
    other = {**identity, 'build': 'changed'}
    write(aot.unit_pack_path('mskf', other), other, [mskf])
    assert Pack(identity={**identity, 'build': 'third'}).reason is not None
    with pytest.raises(ValueError):
        aot.unit_pack_path('unlisted', identity)


def test_mskf_catalog_is_the_forecast_loader_request(tmp_path):
    """The pack key is the exact request woof.core.mskf hands NVRTC, both tiers."""
    pytest.importorskip('cupy')
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tool = root / 'tools' / 'build_gfdleta_pack.py'
    spec = importlib.util.spec_from_file_location('build_unit_pack', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FAMILIES['mskf'][1] == (('mskf', ()), ('mskf', (('GPUWM_MSKF_NO_AEROSOL', 1),)))
    out = tmp_path / 'mskf.json.xz'
    env = dict(os.environ, GPUWM_NUMERICS='fast', GPUWM_WRF_EXACT='0')
    subprocess.run([sys.executable, str(tool), '--work', str(tmp_path), '--catalog', str(out),
                    '--family', 'mskf'], env=env, check=True, cwd=root)
    from woof.kernel_pack import read
    from woof.kernel_aot import compiler_identity, recipe
    rows = [recipe(entry) for entry in read(out, compiler_identity())]
    assert len(rows) == 2 and all(r['compile_request']['filename'] == 'mskf.cu' for r in rows)
    off = [r for r in rows if r['source'].startswith('#define GPUWM_MSKF_NO_AEROSOL 1\n')]
    on = [r for r in rows if r not in off]
    assert len(off) == 1 and len(on) == 1
    assert all('GPUWM_MSKF_FAST' in r['source'] for r in rows)
    # Native operation order protects aerosol numbers only (woof/core/kernels/no_ftz.py).
    assert '-DGPUWM_MSKF_OPERATION_ORDER=1' in on[0]['options']
    assert '-DGPUWM_MSKF_OPERATION_ORDER=1' not in off[0]['options']


def test_shipped_unit_packs_hold_every_current_loader_request(tmp_path):
    """Every shipped unit pack serves the exact request the loader makes today.

    The breakage this prevents (SKEPTIC-3 D, 2026-11-02): a comment-only
    hygiene edit to glibc_flt32.cuh (51bb93111ee) changed the source text of
    every GFDL unit after its packs were built, so no shipped gfdleta image
    matched. A cold stock ra99 forecast then compiled gfdleta_column inside
    step 1 on one compile worker for 27+ min, and the strict replay compiled
    it in-process for 1981 s. A kernel source edit must rebuild the unit
    packs (tools/build_gfdleta_pack.py) in the same change: every catalog
    request (fast, and strict for each referee glibc generation) on every
    shipped architecture, in every compiler's file.
    """
    pytest.importorskip('cupy')
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path
    from woof import kernel_aot as aot
    from woof.kernel_pack import compiler_of, read
    root = Path(__file__).resolve().parents[1]
    tool = root / 'tools' / 'build_gfdleta_pack.py'
    spec = importlib.util.spec_from_file_location('build_unit_pack', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = aot.DEFAULT_PACK.parent / 'kernel-packs'
    missing = []
    for family in aot.UNIT_PACK_FAMILIES:
        wanted = {}
        for label, settings in module.CATALOGS:
            out = tmp_path / f'{family}-{label}.json.xz'
            env = {key: value for key, value in os.environ.items()
                   if key not in ('GPUWM_WRF_EXACT', 'GPUWM_NUMERICS', 'GPUWM_WRF_GLIBC')}
            env.update(settings)
            subprocess.run([sys.executable, str(tool), '--work', str(tmp_path), '--catalog',
                            str(out), '--family', family], env=env, check=True, cwd=root)
            for entry in read(out, compiler_of(out)):
                request = aot.recipe(entry)
                wanted[(request['module_key'], tuple(request['options']))] = (
                    label, request['compile_request'])
        files = sorted(directory.glob(family + '-*.json.xz'))
        assert files, f'no shipped {family} unit pack'
        for path in files:
            held = {(r['module_key'], tuple(r['options']), r['arch'],
                     json.dumps(r['compile_request'], sort_keys=True))
                    for r in read(path, compiler_of(path))}
            for (key, options), (label, metadata) in wanted.items():
                for arch in aot.ARCHITECTURES:
                    if (key, options, arch, json.dumps(metadata, sort_keys=True)) not in held:
                        missing.append(f'{path.name}: {label} {metadata["filename"]} {arch}')
    assert not missing, ('stale unit pack; rebuild with tools/build_gfdleta_pack.py:\n'
                         + '\n'.join(missing))


def test_shipped_images_apply_beside_unnamed_cuda_library_headers(tmp_path, monkeypatch):
    """2.9.12 R11: cupy[ctk] puts cuBLAS/cuFFT/... headers into the toolkit include folder."""
    from woof import kernel_dependencies
    from woof.kernel_dependencies import include_tree_dependencies
    root = tmp_path / 'include'
    root.mkdir()
    (root / 'cuda_fp16.h').write_text('#define HALF 1\n')
    source = '#include <cuda_fp16.h>\n'
    options = ['-I' + str(root)]
    owned = set()
    monkeypatch.setattr(kernel_dependencies, 'library_header_files', lambda: frozenset(owned))
    # Built where the folder holds compiler headers only: same value as the whole-folder binding.
    built = include_tree_dependencies(options, {}, source=source)
    assert built == include_tree_dependencies(options, {})
    bound = dict(row(), source=source, source_sha256=source_hash(source), options=options,
                 include_trees=built)
    pack(tmp_path, monkeypatch, [bound])
    def select(text=source):
        return Pack(tmp_path / 'pack.json', identity={'compiler': 'test'}).select(
            'hexcore.test', text, options, 'sm_120')[1]
    library = root / 'cublas_v2.h'
    library.write_text('#include "cublas_api.h"\n')
    (root / 'cublas_api.h').write_text('int cublas;\n')
    owned.update({library.resolve(), (root / 'cublas_api.h').resolve()})
    assert select() is None
    # Named by the kernel source: bound, so the image misses.
    named = '#if __has_include(<cublas_v2.h>)\n#endif\n' + source
    bound2 = dict(bound, source=named, source_sha256=source_hash(named))
    pack(tmp_path, monkeypatch, [bound2])
    assert select(named) == 'header_tree_mismatch'
    pack(tmp_path, monkeypatch, [bound])
    # Named by a bound compiler header (directly or through another library header): misses.
    (root / 'cuda_fp16.h').write_text('#define HALF 1\n// see cublas_api.h\n')
    assert select() == 'header_tree_mismatch'
    (root / 'cuda_fp16.h').write_text('#define HALF 1\n')
    assert select() is None
    # A changed compiler header or an added header no library owns still misses.
    (root / 'new.h').write_text('#define ADDED 1\n')
    assert select() == 'header_tree_mismatch'
