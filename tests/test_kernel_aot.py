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
