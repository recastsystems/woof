"""Persisted images must be exact, recoverable and safe across shape rebinding."""
import json
import pytest
from woof.kernel_warm_pack import WarmImages


def test_exact_request_survives_process_and_changed_requests_miss(tmp_path):
    args = dict(source="s", options=["--fmad=false"], filename="a.cu", name_expressions=["k<float>"])
    result = b"\x7fELFactual-compiler-result", {"k<float>": "symbol"}
    pack = WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "120")
    pack.put(args, result)
    pack.close()
    second = WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "120")
    assert second.get(args) == result
    assert second.get(dict(args, options=["--fmad=true"])) is None
    assert second.get(dict(args, filename="b.cu")) is None
    assert WarmImages(tmp_path, {"compiler": "cu13"}, "shape/options/strict", "120").get(args) is None
    assert WarmImages(tmp_path, {"compiler": "cu12"}, "other-shape", "120").get(args) == result
    assert WarmImages(tmp_path, {"compiler": "cu12"}, "shape/options/strict", "89").get(args) is None
    # Returned mappings cannot mutate the retained symbol aliases.
    second.get(args)[1]["k<float>"] = "wrong"
    assert second.get(args) == result


def test_damaged_image_and_mapping_fall_back(tmp_path):
    args = dict(source="s")
    pack = WarmImages(tmp_path, {}, "shape", "120")
    pack.put(args, (b"\x7fELFresult", {"alias": "symbol"}))
    pack.close()
    document = json.loads(pack.index.read_text())
    row = next(iter(document["entries"].values()))
    (pack.root / (row["image"] + ".cubin")).write_bytes(b"wrong")
    with pytest.warns(RuntimeWarning):
        assert WarmImages(tmp_path, {}, "shape", "120").get(args) is None
    # Rebuild repairs the bad blob; a corrupted mapping must still miss.
    pack = WarmImages(tmp_path, {}, "shape", "120")
    pack.put(args, (b"\x7fELFresult", {"alias": "symbol"}))
    pack.close()
    row["mapping"]["alias"] = "wrong"
    pack.index.write_text(json.dumps(document))
    with pytest.warns(RuntimeWarning):
        assert WarmImages(tmp_path, {}, "shape", "120").get(args) is None


def test_parallel_publish_keeps_both_members(tmp_path):
    one = WarmImages(tmp_path, {}, "shape", "120")
    two = WarmImages(tmp_path, {}, "shape", "120")
    one.put({"source": "one"}, (b"\x7fELFone", None))
    two.put({"source": "two"}, (b"\x7fELFtwo", None))
    one.close()
    two.close()
    loaded = WarmImages(tmp_path, {}, "shape", "120")
    assert loaded.get({"source": "one"}) == (b"\x7fELFone", None)
    assert loaded.get({"source": "two"}) == (b"\x7fELFtwo", None)


def test_metadata_second_process_identity_options_and_corruption(tmp_path):
    from woof.compiler_metadata import CompilerMetadata
    args = dict(kind="empty-preprocess", options=["--ftz=false"], arch="120")
    measured = []
    def measure():
        measured.append(1)
        return "// measured compiler build"
    cache = CompilerMetadata(tmp_path, {"library": "cu12-bytes"})
    assert cache.resolve(args, measure) == "// measured compiler build"
    second = CompilerMetadata(tmp_path, {"library": "cu12-bytes"})
    assert second.resolve(args, measure) == "// measured compiler build"
    assert len(measured) == 1
    CompilerMetadata(tmp_path, {"library": "cu13-bytes"}).resolve(args, measure)
    second.resolve(dict(args, options=["--ftz=true"]), measure)
    second.resolve(dict(args, arch="89"), measure)
    assert len(measured) == 4
    path = next(cache.root.glob("*.json"))
    row = json.loads(path.read_text());row['value'] = 'corrupted'
    path.write_text(json.dumps(row))
    with pytest.warns(RuntimeWarning):
        CompilerMetadata(tmp_path, {"library": "cu12-bytes"}).resolve(row['arguments'], measure)
    assert len(measured) == 5


def test_metadata_single_flight_and_retry(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from woof.compiler_metadata import CompilerMetadata
    cache = CompilerMetadata(tmp_path, {})
    entered,release = threading.Event(),threading.Event()
    calls = []
    def measure():
        calls.append(1);entered.set();assert release.wait(5)
        return "// actual output"
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.resolve, {}, measure)
        assert entered.wait(5)
        second = pool.submit(cache.resolve, {}, measure)
        release.set()
        assert first.result() == second.result() == "// actual output"
    assert calls == [1]
    def fail(): raise RuntimeError('compiler refused')
    with pytest.raises(RuntimeError, match='compiler refused'):
        cache.resolve({'retry': True}, fail)
    assert cache.resolve({'retry': True}, lambda: '// recovered') == '// recovered'


def test_metadata_bridge_leaves_nonempty_and_other_backend_compilation_intact(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof.kernel_warm_pack import install_metadata
    from woof.certify import compile_platform
    calls = []
    banners = []
    def preprocess(*args):
        calls.append(args)
        return '// measured preprocess'
    def banner():
        banners.append(1)
        return '// measured banner'
    compiler = SimpleNamespace(_preprocess=preprocess)
    monkeypatch.setattr(compile_platform, 'nvrtc_banner', banner)
    monkeypatch.setattr('woof.kernel_aot.effective_options', tuple)
    install_metadata(compiler, tmp_path, {'library': 'actual-digest'})
    assert compiler._preprocess('', ('-O3',), '120', 'nvrtc') == '// measured preprocess'
    assert compiler._preprocess('', ('-O3',), '120', 'nvrtc') == '// measured preprocess'
    compiler._preprocess('real source', ('-O3',), '120', 'nvrtc')
    compiler._preprocess('', ('-O3',), '120', 'nvcc')
    assert len(calls) == 3
    # install_metadata only wraps CuPy's empty preprocess route.
    assert compile_platform.nvrtc_banner is banner


def test_banner_persists_measured_bytes_and_refuses_unresolved_identity(tmp_path):
    from types import SimpleNamespace
    from woof.compiler_metadata import nvrtc_banner
    calls = []
    nvrtc = SimpleNamespace(createProgram=lambda *a: 1,
        compileProgram=lambda *a: calls.append(a), destroyProgram=lambda *a: None,
        getPTX=lambda *a: b'// Compiler Build ID: CL-123\n// Cuda compilation tools, release 13.4, V13.4.92\n.version 8.8\n')
    library = dict(status='resolved', sha256='loaded-compiler-bytes')
    first = nvrtc_banner(nvrtc, library, tmp_path)
    assert first == nvrtc_banner(nvrtc, library, tmp_path)
    assert len(calls) == 1 and '.version' not in first
    nvrtc_banner(nvrtc, dict(library, sha256='changed-compiler'), tmp_path)
    nvrtc_banner(nvrtc, dict(status='unavailable'), tmp_path)
    nvrtc_banner(nvrtc, dict(status='unavailable'), tmp_path)
    assert len(calls) == 4


def test_literal_and_nested_headers_bind_actual_bytes(tmp_path):
    headers = tmp_path / 'include'
    headers.mkdir()
    header = headers / 'one.h'
    nested = headers / 'two.h'
    header.write_text('#include "two.h"\n')
    nested.write_text('#define VALUE 1\n')
    args = dict(source='#include "one.h"\n', options=['-I'+str(headers)])
    one = WarmImages(tmp_path / 'packs', {}, 'shape', '120')
    one.put(args, (b'\x7fELFheader-one', None));one.close()
    assert WarmImages(tmp_path / 'packs', {}, 'shape', '120').get(args)[0] == b'\x7fELFheader-one'
    nested.write_text('#define VALUE 2\n')
    assert WarmImages(tmp_path / 'packs', {}, 'shape', '120').get(args) is None


def test_macro_header_refuses_old_image_and_storing_unbound_result(tmp_path):
    args = dict(source='#define H "owned.h"\n#include H\n', options=[])
    pack = WarmImages(tmp_path, {}, 'shape', '120')
    with pytest.warns(RuntimeWarning, match='dependency is unbound'):
        pack.put(args, (b'\x7fELFstale-header', None))
    assert pack.get(args) is None and not pack.rows


def test_unbound_requests_and_shared_header_trees_are_checked_once(tmp_path, monkeypatch):
    headers = tmp_path / 'include'
    headers.mkdir()
    header = headers / 'host.h'
    header.write_text('#include "../outside.h"\n')
    from pathlib import Path
    original = Path.read_bytes
    reads = []
    def read(path):
        if path == header:
            reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    images = WarmImages(tmp_path / 'cache', {}, 'shape', '120')
    args = dict(source='one', options=['-I' + str(headers)])
    with pytest.warns(RuntimeWarning, match='may escape'):
        assert images.get(args) is None
    images.put(args, (b'\x7fELFunbound', None))
    assert images.get(dict(args, source='two')) is None
    assert len(reads) == 1
    assert not images.rows and not images.pending
    # A new session checks the actual bytes again, including repaired roots.
    header.write_text('// no external dependency\n')
    assert WarmImages(tmp_path / 'cache', {}, 'shape', '120')._request_key(args) is not None
    assert len(reads) == 2


def test_header_escape_snapshot_binds_the_complete_search_roots(tmp_path):
    from woof.kernel_dependencies import include_tree_dependencies
    headers = tmp_path / 'include'
    headers.mkdir()
    (headers / 'host.h').write_text('#include "../outside.h"\n')
    (tmp_path / 'outside.h').write_text('// included from parent\n')
    snapshots = {}
    narrow = ['-I' + str(headers)]
    wide = [*narrow, '-I' + str(tmp_path)]
    with pytest.raises(ValueError, match='may escape'):
        include_tree_dependencies(narrow, snapshots)
    assert len(include_tree_dependencies(wide, snapshots)) == 2
    with pytest.raises(ValueError, match='may escape'):
        include_tree_dependencies(narrow, snapshots)


def test_escaped_windows_linker_string_is_not_an_absolute_include(tmp_path):
    from woof.kernel_dependencies import include_tree_dependencies
    (tmp_path / 'nvtx.h').write_text(
        r'#define EXPORT _Pragma("comment(linker, \"/EXPORT:\" __FUNCTION__)")' + '\n')
    assert len(include_tree_dependencies(['-I' + str(tmp_path)], {})) == 1


def test_header_parser_does_not_hide_directive_after_comment_like_string(tmp_path):
    from woof.kernel_dependencies import header_dependencies
    header = tmp_path / 'owned.h';header.write_text('#define VALUE 1\n')
    source = 'const char* a="/*";\n/* comment */ #include "owned.h"\nconst char* b="*/";'
    assert header_dependencies(source, ['-I'+str(tmp_path)])[0]['path'] == str(header)


def test_forced_include_metadata_uses_original_probe(tmp_path):
    from types import SimpleNamespace
    from woof.kernel_warm_pack import install_metadata
    calls = []
    compiler = SimpleNamespace(_preprocess=lambda *args: calls.append(args) or 'actual')
    install_metadata(compiler, tmp_path, {})
    for _ in range(2):
        with pytest.warns(RuntimeWarning, match='forced-include'):
            compiler._preprocess('', ('--pre-include=owned.h',), '120', 'nvrtc')
    assert len(calls) == 2


def test_header_comment_replacement_preserves_directive_boundaries(tmp_path):
    from woof.kernel_dependencies import header_dependencies
    header = tmp_path / 'owned.h'
    header.write_text('#define VALUE 1\n')
    options = ['-I' + str(tmp_path)]
    source = 'const char* s="// /* \\"quoted\\" */";\n/*before\n comment*/ #include /*between*/ "owned.h" //after\n'
    assert header_dependencies(source, options)[0]['path'] == str(header)
    # A null directive cannot consume the following non-directive line.
    assert header_dependencies('#\ninclude "missing.h"\n', options) == []
    with pytest.raises(ValueError, match='macro or nonliteral'):
        header_dependencies('/* a\n b */ #include NAME\n', options)


def test_content_pack_moves_installation_and_run_tag(tmp_path):
    from woof.kernel_warm_pack import pack, seed
    first = tmp_path / 'box-one' / 'include'
    second = tmp_path / 'box-two' / 'include'
    for root in (first, second):
        root.mkdir(parents=True)
        (root / 'value.h').write_text('#define VALUE 7\n')
    args = dict(source='#include "value.h"\n', options=['-I'+str(first)], filename='unit.cu')
    cache = WarmImages(tmp_path / 'cold', {'nvrtc': [13, 4]}, 'cycle-one-shape', '120')
    result = (b'\x7fELFcompiled-unit', None)
    cache.put(args, result); cache.close()
    archive = tmp_path / 'images.tar.gz'
    receipt = pack(archive, root=tmp_path / 'cold', architectures=['sm_120'])
    seed(archive, root=tmp_path / 'seeded', sha256=receipt['sha256'])
    target = WarmImages(tmp_path / 'seeded', {'nvrtc': [13, 4]}, 'other-cycle-shape', 'sm_120')
    relocated = dict(args, options=['-I'+str(second)])
    assert target.get(relocated) == result
    (second / 'value.h').write_text('#define VALUE 8\n')
    assert WarmImages(tmp_path / 'seeded', {'nvrtc': [13, 4]}, '', '120').get(relocated) is None
    assert WarmImages(tmp_path / 'seeded', {'nvrtc': [13, 5]}, '', '120').get(relocated) is None
    archive.write_bytes(archive.read_bytes() + b'damaged')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        seed(archive, root=tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_guarded_inlined_header_and_conditional_discovery(tmp_path):
    args = dict(source='#define GUARD\n#ifndef GUARD\n#include "inlined.h"\n#endif\n'
                '#if __has_include("optional.h")\n#define VALUE 1\n#endif\n', options=['-I'+str(tmp_path)])
    before = WarmImages.key(args)
    (tmp_path / 'optional.h').write_text('// optional exists\n')
    assert WarmImages.key(args) != before


def test_macro_selected_headers_bind_tree_and_relocation(tmp_path):
    roots = [tmp_path / 'one', tmp_path / 'two']
    for root in roots:
        root.mkdir()
        (root / 'chosen.h').write_text('#define VALUE 4\n')
    args = dict(source='#define H "chosen.h"\n#include H\n', options=['-I'+str(roots[0])])
    assert WarmImages.key(args) == WarmImages.key(dict(args, options=['-I'+str(roots[1])]))
    (roots[1] / 'chosen.h').write_text('#define VALUE 5\n')
    assert WarmImages.key(args) != WarmImages.key(dict(args, options=['-I'+str(roots[1])]))


def test_program_names_are_normalized_before_compiling():
    from woof.kernel_warm_pack import program_name
    assert program_name('code', 'a'*40+'.cubin.cu') == program_name('code', 'b'*40+'.cubin.cu')
    assert program_name('different', 'a'*40+'.cubin.cu') != program_name('code', 'a'*40+'.cubin.cu')
    assert program_name('code', 'direct.cu') == 'direct.cu'


def test_seed_merges_local_images_and_rejects_corrupt_index(tmp_path):
    from woof.kernel_warm_pack import pack, seed
    source = WarmImages(tmp_path / 'source', {}, None, '89')
    source.put({'source': 'remote'}, (b'\x7fELFremote', None)); source.close()
    local = WarmImages(tmp_path / 'target', {}, None, '89')
    local.put({'source': 'local'}, (b'\x7fELFlocal', None)); local.close()
    archive = tmp_path / 'pack.tar.gz'
    pack(archive, root=tmp_path / 'source')
    seed(archive, root=tmp_path / 'target')
    target = WarmImages(tmp_path / 'target', {}, None, '89')
    assert target.get({'source': 'local'})[0] == b'\x7fELFlocal'
    assert target.get({'source': 'remote'})[0] == b'\x7fELFremote'
    document = json.loads(source.index.read_text())
    next(iter(document['entries'].values()))['mapping'] = {'bad': 'alias'}
    source.index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='entry digest mismatch'):
        pack(tmp_path / 'bad.tar.gz', root=tmp_path / 'source')


def test_cache_environment_and_arch_binding(tmp_path, monkeypatch):
    from woof.kernel_warm_pack import cache_root
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    monkeypatch.delenv('WOOF_KERNEL_CACHE', raising=False)
    assert cache_root() == tmp_path / 'xdg/woof/kernels'
    monkeypatch.setenv('WOOF_KERNEL_CACHE', str(tmp_path / 'explicit'))
    assert cache_root() == tmp_path / 'explicit'
    images = WarmImages(cache_root(), {}, None, '120')
    images.put({'source': 'code', 'arch': 'device'}, (b'\x7fELFcode', None))
    assert images.get({'source': 'code', 'arch': '120'})[0] == b'\x7fELFcode'
    assert images.get({'source': 'code', 'arch': '89'}) is None
    images.close()
