"""Exact binary artifact reuse and invalidation, without a GPU."""
import json
import struct
import pytest
from woof.hex.kernel_warm import Store


def test_member_local_caches_share_kernel_artifacts(tmp_path, monkeypatch):
    from woof.hex import kernel_warm
    monkeypatch.delenv('WOOF_HEX_KERNEL_CACHE_ROOT', raising=False)
    monkeypatch.delenv('WOOF_HEX_CACHE_ROOT', raising=False)
    monkeypatch.delenv(kernel_warm.DIRECTORY_ENVIRONMENT, raising=False)
    monkeypatch.setenv('WOOF_HEX_KERNEL_PREWARM', 'on')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'box'))
    identity = dict(nvrtc=[13, 2], library='compiler')
    args = dict(source='kernel', options=[], arch='120')
    roots = []
    for member in ('cycle-a/member1', 'cycle-b/member2'):
        monkeypatch.setenv('WOOF_HEX_PREP_CACHE', str(tmp_path / member / 'prep'))
        monkeypatch.setenv('CUPY_CACHE_DIR', str(tmp_path / member / 'cupy'))
        root = kernel_warm._root()
        roots.append(root)
        store = Store(root, identity, member, workers=1)
        compile_fn = (lambda: (b'image', None)) if len(roots) == 1 else lambda: pytest.fail('member recompiled')
        assert store.call('nvrtc', args, compile_fn) == (b'image', None)
        store.pool.shutdown()
    assert roots == [tmp_path / 'box/woof hex/prep/kernel-packs'] * 2
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT', str(tmp_path / 'application'))
    assert kernel_warm.cache_root() == tmp_path / 'application'
    monkeypatch.setenv('WOOF_HEX_KERNEL_CACHE_ROOT', str(tmp_path / 'kernels'))
    assert kernel_warm.cache_root() == tmp_path / 'kernels'


def test_fresh_process_store_returns_identical_image_and_mapping(tmp_path):
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    first = Store(tmp_path, identity, "shape=163842x55", workers=2)
    args = dict(source="kernel", options=["--fmad=false"], arch="120", filename="k.cu")
    expected = b"\x7fELF\x00image", {"f<float>": "_f_float"}
    assert first.call("nvrtc", args, lambda: expected) == expected
    first.flush()
    first.pool.shutdown()
    second = Store(tmp_path, identity, "shape=163842x55", workers=2)
    def forbidden():
        pytest.fail("warm request compiled")
    result = second.call("nvrtc", args, forbidden)
    assert result == expected
    result[1]["f<float>"] = "edited by caller"
    assert second.call("nvrtc", args, forbidden) == expected
    assert second.receipt()["hits"] == 2
    second.pool.shutdown()


def test_a_new_run_configuration_reuses_the_content_addressed_image(tmp_path):
    """THE BREAKAGE THIS PREVENTS: artifacts were filed under the run's argv, so every new storm-following member
    (a new --mesh/--grid) recompiled the MPAS suite's fast kernels inline, 15 s a moving segment
    (MPAS-ROUTE-SPEED-2026-10-24).  The key binds the whole compiler request, so another run shares it."""
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    args = dict(source="kernel", options=["--fmad=false"], arch="120", filename="k.cu")
    first = Store(tmp_path, identity, '{"argv": ["forecast", "--mesh", "start"]}', workers=2)
    first.call("nvrtc", args, lambda: (b"image", None))
    first.flush()
    first.pool.shutdown()
    second = Store(tmp_path, identity, '{"argv": ["forecast", "--mesh", "mn02900e26450"]}', workers=2)
    assert second.call("nvrtc", args, lambda: pytest.fail("a new run recompiled a cached request")) == (b"image", None)
    second.pool.shutdown()


@pytest.mark.parametrize("change", ["options", "source", "arch", "filename", "headers", "compiler"])
def test_changed_compiler_inputs_cannot_serve_stale_image(tmp_path, change):
    include = tmp_path / "include"
    include.mkdir()
    header = include / "config.h"
    header.write_text("#define VALUE 1\n")
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    args = dict(source='#include "config.h"', options=["-I" + str(include), "--fmad=false"],
                arch="120", filename="k.cu")
    first = Store(tmp_path / "cache", identity, "shape-a", workers=2)
    first.call("nvrtc", args, lambda: (b"old", None))
    first.flush()
    first.pool.shutdown()
    next_args = dict(args)
    next_identity = dict(identity)
    shape = "shape-a"
    if change == "headers":
        old_stat = header.stat()
        header.write_text("#define VALUE 2\n")
        import os
        os.utime(header, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    elif change == "compiler":
        next_identity["library"] = "compiler-b"
    elif change == "options":
        next_args["options"] = ["-I" + str(include), "--fmad=true"]
    else:
        next_args[change] += "changed"
    second = Store(tmp_path / "cache", next_identity, shape, workers=2)
    assert second.call("nvrtc", next_args, lambda: (b"new", None)) == (b"new", None)
    second.flush()
    second.pool.shutdown()


@pytest.mark.parametrize("damage", ["payload", "mapping", "truncated", "header_type"])
def test_corrupt_artifact_recompiles_instead_of_loading(tmp_path, damage):
    identity = dict(nvrtc=[12, 9])
    first = Store(tmp_path, identity, "shape", workers=2)
    args = dict(source="kernel", options=[], arch="120")
    key = first.request("nvrtc", args)
    first.put(key, (b"image", {"k": "symbol"}))
    first.flush()
    first.pool.shutdown()
    path = first.path(key)
    data = path.read_bytes()
    if damage == "truncated":
        data = data[:9]
    elif damage == "header_type":
        data = data[:8] + struct.pack("<I", 2) + b"[]"
    elif damage == "mapping":
        data = data.replace(b"symbol", b"broken")
    else:
        data = data[:-1] + b"!"
    path.write_bytes(data)
    second = Store(tmp_path, identity, "shape", workers=2)
    assert second.call("nvrtc", args, lambda: (b"recompiled", None)) == (b"recompiled", None)
    assert second.receipt()["invalid"] == 2  # initial read and recheck under the process lock
    second.flush()
    second.pool.shutdown()


def test_cuda12_and_cuda13_coexist_and_text_probes_are_persisted(tmp_path):
    for major in (12, 13):
        identity = dict(nvrtc=[major, 4])
        first = Store(tmp_path, identity, "shape", workers=2)
        first.call("preprocess", {"options": [], "arch": "120"}, lambda: "// banner" + str(major))
        first.flush()
        first.pool.shutdown()
    for major in (12, 13):
        second = Store(tmp_path, dict(nvrtc=[major, 4]), "shape", workers=2)
        assert second.call("preprocess", {"options": [], "arch": "120"},
                           lambda: pytest.fail("warm probe compiled")) == "// banner" + str(major)
        second.pool.shutdown()


def test_symlink_include_directory_uses_compiler(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "value.h").write_text("#define VALUE 1\n")
    (include / "linked").symlink_to(target, target_is_directory=True)
    store = Store(tmp_path / "cache", dict(nvrtc=[13, 4]), "shape", workers=2)
    args = dict(source='#include <linked/value.h>', options=["-I" + str(include)])
    assert store.call("nvrtc", args, lambda: (b"first", None)) == (b"first", None)
    assert store.call("nvrtc", args, lambda: (b"second", None)) == (b"second", None)
    assert store.receipt()["bypasses"] == 2
    store.pool.shutdown()


def test_nested_header_escaping_include_tree_uses_compiler(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "config.h").write_text('#include "../outside.h"\n')
    (tmp_path / "outside.h").write_text("#define VALUE 1\n")
    store = Store(tmp_path / "cache", dict(nvrtc=[13, 4]), "shape", workers=2)
    args = dict(source='#include <config.h>', options=["-I" + str(include)])
    assert store.call("nvrtc", args, lambda: (b"first", None)) == (b"first", None)
    (tmp_path / "outside.h").write_text("#define VALUE 2\n")
    assert store.call("nvrtc", args, lambda: (b"second", None)) == (b"second", None)
    assert store.receipt()["bypasses"] == 2
    store.pool.shutdown()


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("broad_first", [False, True])
def test_include_root_metadata_does_not_reuse_another_requests_boundary(tmp_path, absolute, broad_first):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    header = b / "value.h"
    header.write_text("#define VALUE 1\n")
    operand = str(header) if absolute else "../b/value.h"
    (a / "wrapper.h").write_text('#include "' + operand + '"\n')
    args = dict(source="#include <wrapper.h>", options=["-I" + str(a)])
    broad = dict(args, options=args["options"] + ["-I" + str(b)])
    store = Store(tmp_path / "cache", dict(nvrtc=[13, 4]), "shape", workers=2)
    if broad_first:
        store.request("nvrtc", broad)
    with pytest.warns(UserWarning, match="escapes the bound header trees"):
        assert store.call("nvrtc", args, lambda: (b"narrow-first", None)) == (b"narrow-first", None)
    store.request("nvrtc", broad)
    assert store.call("nvrtc", args, lambda: (b"narrow-second", None)) == (b"narrow-second", None)
    assert store.receipt()["bypasses"] == 2
    assert store.call("nvrtc", broad, lambda: (b"bounded", None)) == (b"bounded", None)
    assert store.call("nvrtc", broad, lambda: pytest.fail("bounded request recompiled")) == (b"bounded", None)
    store.flush()
    store.pool.shutdown()


def test_same_header_closure_is_shared_but_executable_keys_are_distinct(tmp_path, monkeypatch):
    import concurrent.futures
    import threading
    include = tmp_path / 'include'
    include.mkdir()
    (include / 'value.h').write_text('#define VALUE 1\n')
    store = Store(tmp_path / 'cache', dict(nvrtc=[13, 4]), 'shape', workers=2)
    entered, release = threading.Event(), threading.Event()
    original = store._bind_includes
    calls = []
    def delayed(*args):
        calls.append(args)
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(store, '_bind_includes', delayed)
    try:
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(store.request, 'nvrtc', dict(
                source='#include <value.h>\nint kernel_' + str(i) + ';',
                options=['-I' + str(include)])) for i in range(24)]
            assert entered.wait(5)
            release.set()
            keys = [f.result(timeout=5) for f in futures]
        assert len(calls) == 1
        assert len(set(keys)) == 24
        roots = store._includes(['-I' + str(include)], '#include <value.h>')
        roots.clear()
        assert len(store._includes(['-I' + str(include)], '#include <value.h>')) == 1
    finally:
        release.set()
        store.pool.shutdown()


@pytest.mark.parametrize('definition', ['source', 'option'])
def test_shared_closure_revalidates_changed_macro_inputs(tmp_path, definition):
    include = tmp_path / 'include'
    include.mkdir()
    (include / 'wrapper.h').write_text('#include CHOICE\n')
    (include / 'safe.h').write_text('#define VALUE 1\n')
    outside = tmp_path / 'outside.h'
    outside.write_text('#define VALUE 2\n')
    store = Store(tmp_path / 'cache', dict(nvrtc=[13, 4]), 'shape', workers=2)
    def args(value):
        source, options = '#include <wrapper.h>', ['-I' + str(include)]
        if definition == 'source':
            source = '#define CHOICE "' + value + '"\n' + source
        else:
            options.append('-DCHOICE="' + value + '"')
        return dict(source=source, options=options)
    try:
        safe = store.request('nvrtc', args('safe.h'))
        with pytest.raises(ValueError, match='escapes the bound header trees'):
            store.request('nvrtc', args(str(outside)))
        assert store.request('nvrtc', args('safe.h')) == safe
    finally:
        store.pool.shutdown()


def test_failed_closure_is_not_memoized(tmp_path, monkeypatch):
    store = Store(tmp_path, dict(nvrtc=[13, 4]), 'shape', workers=2)
    original = store._bind_includes
    attempts = []
    def once(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError('temporary read failure')
        return original(*args)
    monkeypatch.setattr(store, '_bind_includes', once)
    try:
        with pytest.raises(OSError, match='temporary read failure'):
            store.request('nvrtc', dict(source='kernel'))
        store.request('nvrtc', dict(source='kernel'))
        assert len(attempts) == 2
    finally:
        store.pool.shutdown()


def test_failed_banner_probe_is_retried(tmp_path):
    store = Store(tmp_path, dict(nvrtc=[13, 4]), "shape", workers=2)
    assert store.call("compiler-banner", {}, lambda: "") == ""
    assert store.call("compiler-banner", {}, lambda: "// measured build") == "// measured build"
    store.flush()
    assert store.receipt()["writes"] == 1
    store.pool.shutdown()


@pytest.mark.parametrize("style", ["source", "option", "nested", "continued"])
def test_macro_include_header_bytes_invalidate_with_same_timestamp(tmp_path, style):
    import os
    include = tmp_path / "headers"
    include.mkdir()
    header = include / "value.h"
    header.write_text("#define VALUE 1\n")
    definition = '"' + str(header) + '"'
    options = []
    source = '#define HEADER ' + definition + '\n#include HEADER\n'
    if style == "option":
        options = ['-DHEADER=' + definition]
        source = '#include HEADER\n'
    elif style == "nested":
        (include / "wrapper.h").write_text(source)
        options = ['-I' + str(include)]
        source = '#include <wrapper.h>\n'
    elif style == "continued":
        source = '#define HEADER \\\n' + definition + '\n# /* comment */ include HEADER // tail\n'
    args = dict(source=source, options=options)
    def invoke(expected):
        store = Store(tmp_path / 'cache', dict(nvrtc=[13, 4]), 'shape', workers=2)
        result = store.call('nvrtc', args, lambda: (expected, None))
        store.flush()
        receipt = store.receipt()
        store.pool.shutdown()
        return result, receipt
    assert invoke(b'old')[0] == (b'old', None)
    assert invoke(b'wrong')[0] == (b'old', None)
    stat = header.stat()
    header.write_text("#define VALUE 2\n")
    os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed, receipt = invoke(b'new')
    assert changed == (b'new', None)
    assert receipt['misses'] == 1 and receipt['bypasses'] == 0
    assert invoke(b'wrong')[0] == (b'new', None)


@pytest.mark.parametrize('operand,definition', [
    ('HEADER', '#define HEADER "../outside.h"\n'),
    ('HEADER(value)', '#define HEADER(x) #x\n'),
    ('UNKNOWN_HEADER', ''),
])
def test_unbound_macro_include_warns_and_retains_compilation(tmp_path, operand, definition):
    include = tmp_path / 'headers'
    include.mkdir()
    (include / 'wrapper.h').write_text(definition + '#include ' + operand + '\n')
    (tmp_path / 'outside.h').write_text('#define VALUE 1\n')
    args = dict(source='#include <wrapper.h>', options=['-I' + str(include)])
    store = Store(tmp_path / 'cache', dict(nvrtc=[13, 4]), 'shape', workers=2)
    with pytest.warns(UserWarning, match='ordinary compilation retained'):
        assert store.call('nvrtc', args, lambda: (b'first', None)) == (b'first', None)
    assert store.call('nvrtc', args, lambda: (b'second', None)) == (b'second', None)
    assert store.receipt()['bypasses'] == 2
    assert store.receipt()['writes'] == 0
    store.pool.shutdown()


def test_header_directives_preserve_lines_strings_and_comment_boundaries():
    from woof.hex.kernel_warm import _include_metadata, _expanded_headers
    source = '''/* a comment
       #include "not-a-header.h"
    */
#define HEADER /* retained token boundary */ "real.h"
#define UNUSED "literal/*string*/.h"
# /* before directive */ include HEADER // trailing comment
#define FUNCTION(x) x
'''
    definitions, includes = _include_metadata(source)
    assert includes == ['HEADER']
    assert _expanded_headers(includes[0], definitions) == ['real.h']
    assert definitions['UNUSED'] == {'"literal/*string*/.h"'}
    assert definitions['FUNCTION'] == {None}
    assert _include_metadata('#\ninclude "not-a-directive.h"\n')[1] == []


def _settled_header_tree(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "config.h").write_text("#define VALUE 1\n")
    (include / "other.h").write_text('#include "config.h"\n#define OTHER VALUE\n')
    return include, dict(source='#include "config.h"\n', options=["-I" + str(include)], arch="120",
                         filename="k.cu")


def test_settled_header_tree_is_not_reparsed_by_a_new_process(tmp_path, monkeypatch):
    """THE COST THIS REMOVES: every forecast process read and regex-parsed every file of every bound include tree
    (thousands of CUDA/CCCL headers) on the GIL, 4-18 s that starved the main thread's host preparation.  A tree
    whose files all keep their size, inode, mtime and ctime binds from the stored result: the same request key."""
    import time
    include, args = _settled_header_tree(tmp_path)
    real = time.time_ns
    monkeypatch.setattr(time, "time_ns", lambda: real() + 5_000_000_000)   # the files are seconds old
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    first = Store(tmp_path / "cache", identity, "shape", workers=2)
    key = first.request("nvrtc", args)
    first.pool.shutdown()
    assert list((tmp_path / "cache" / "header-trees").glob("*.json"))
    second = Store(tmp_path / "cache", identity, "shape", workers=2)
    second.pool.map = lambda *a, **k: pytest.fail("an unchanged settled header tree was parsed again")
    assert second.request("nvrtc", args) == key
    second.pool.shutdown()
    # A file added to the tree changes its listing: the tree is parsed again.
    (include / "added.h").write_text("#define ADDED 1\n")
    third = Store(tmp_path / "cache", identity, "shape", workers=2)
    parsed = []
    real_map = third.pool.map
    third.pool.map = lambda *a, **k: parsed.append(1) or real_map(*a, **k)
    assert third.request("nvrtc", args) != key
    assert parsed
    third.pool.shutdown()


def test_a_tree_changed_within_a_second_of_its_parse_is_never_stored(tmp_path):
    """File times are coarse: an edit in the same clock tick can keep every stamp, so a tree whose newest change
    is under a second old is parsed again by the next process (git's racy-clean rule)."""
    include, args = _settled_header_tree(tmp_path)
    store = Store(tmp_path / "cache", dict(nvrtc=[13, 4]), "shape", workers=2)
    store.request("nvrtc", args)
    store.pool.shutdown()
    assert not list((tmp_path / "cache" / "header-trees").glob("*.json"))


def _concurrent_artifact(root, ready, release, results, source='kernel'):
    import os
    import time
    store = Store(root, dict(nvrtc=[12, 9]), str(os.getpid()), workers=1)
    ready.put(os.getpid())
    release.wait(15)
    def compile_one():
        with (root / 'compiler-calls').open('a') as stream:
            stream.write(source + '\n')
        time.sleep(0.2)
        return b'exact image', {'entry': 'symbol'}
    try:
        value = store.call('nvrtc', dict(source=source, options=[], arch='120'), compile_one)
        results.put((value, store.receipt()))
    finally:
        store.pool.shutdown()


def test_concurrent_processes_publish_once_before_unlocking(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    ready, results, release = context.Queue(), context.Queue(), context.Event()
    processes = [context.Process(target=_concurrent_artifact, args=(tmp_path, ready, release, results))
                 for _ in range(4)]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            ready.get(timeout=20)
        release.set()
        rows = [results.get(timeout=20) for _ in processes]
        assert all(value == (b'exact image', {'entry': 'symbol'}) for value, _ in rows)
        assert (tmp_path / 'compiler-calls').read_text().splitlines() == ['kernel']
        assert sum(row['compiles'] for _, row in rows) == 1
        assert sum(row['shared_hits'] for _, row in rows) == 3
    finally:
        for process in processes:
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)


def test_failed_compile_releases_lock_for_the_next_store(tmp_path):
    first = Store(tmp_path, dict(nvrtc=[12, 9]), 'first', workers=1)
    args = dict(source='kernel', options=[], arch='120')
    def failed():
        raise RuntimeError('compiler refused')
    try:
        with pytest.raises(RuntimeError, match='compiler refused'):
            first.call('nvrtc', args, failed)
        second = Store(tmp_path, first.identity, 'second', workers=1)
        try:
            assert second.call('nvrtc', args, lambda: (b'image', None)) == (b'image', None)
        finally:
            second.pool.shutdown()
    finally:
        first.pool.shutdown()


def test_an_isolated_store_reads_the_box_store_without_writing_to_it(tmp_path):
    """THE BREAKAGE (V8X6000 and an RTX 5090 host, 2026-10-29): a run with its own cache root compiled every module the
    box already held.  It now reads the box's default and image stores (read-only, same request key)."""
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    args = dict(source="kernel", options=["--fmad=false"], arch="120", filename="k.cu")
    box = Store(tmp_path / "box", identity, "first run", workers=1)
    box.call("nvrtc", args, lambda: (b"\x7fELFimage", None))
    box.flush()
    box.pool.shutdown()
    own = Store(tmp_path / "lane", identity, "isolated run", workers=1, fallbacks=[tmp_path / "box"])
    key = own.request("nvrtc", args)
    assert own.held(key) and not own.path(key).exists()
    assert own.call("nvrtc", args, lambda: pytest.fail("recompiled what the box holds")) == (b"\x7fELFimage", None)
    assert own.receipt()["box_hits"] == 1
    other = dict(args, options=["--fmad=true"])
    assert own.call("nvrtc", other, lambda: (b"\x7fELFother", None)) == (b"\x7fELFother", None)
    own.flush()
    assert own.path(own.request("nvrtc", other)).is_file()
    assert not (tmp_path / "box" / own.path(own.request("nvrtc", other)).relative_to(tmp_path / "lane")).exists()
    own.pool.shutdown()


def test_box_roots_are_the_default_store_and_the_image_store(tmp_path, monkeypatch):
    from woof.hex import kernel_warm
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(kernel_warm.SYSTEM_ENVIRONMENT, str(tmp_path / "image"))
    default = tmp_path / "xdg" / "woof hex" / "prep" / "kernel-packs"
    assert kernel_warm.box_roots(tmp_path / "lane") == [default, tmp_path / "image"]
    assert kernel_warm.box_roots(default) == [tmp_path / "image"]
