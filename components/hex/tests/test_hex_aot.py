"""A corrupt or stale image must never become a cached kernel."""
import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from woof.hex.cuda_backend.aot import Pack, source_hash


def row(arch="sm_120"):
    blob = b"\x7fELFtest-image"
    return {"module_key": "hexcore.test", "source": "source", "arch": arch,
            "source_sha256": source_hash("source"), "options": ["--fmad=false"],
            "cubin_base64": base64.b64encode(blob).decode(),
            "cubin_sha256": hashlib.sha256(blob).hexdigest()}


def pack(tmp_path, monkeypatch, rows=None):
    write = pytest.importorskip("woof.hex.kernel_pack").write
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    path = tmp_path / "pack.json"
    write(path, {"compiler": "test"}, rows or [row()])
    return Pack(path, identity={"compiler": "test"})


def test_engine_without_numerics_keeps_its_original_options(monkeypatch):
    import sys
    from woof.hex.cuda_backend.aot import effective_options, preserve_final_options
    monkeypatch.setitem(sys.modules, "woof.numerics", None)
    assert effective_options(["--fmad=false"]) == ("--fmad=false",)
    with preserve_final_options():
        assert effective_options(["--fmad=false"]) == ("--fmad=false",)


def test_cpu_compiler_preserves_final_options_in_fast_environment(monkeypatch):
    from woof import numerics
    from woof.hex.cuda_backend.aot import preserve_final_options

    monkeypatch.setattr(numerics, "_mode", "fast")
    strict, fast = ("--fmad=false",), ("--fmad=true",)
    assert numerics.effective_options(strict) == ()
    with preserve_final_options():
        assert numerics.effective_options(strict) == strict
        assert numerics.effective_options(fast) == fast
    assert numerics.effective_options(strict) == ()
    with pytest.raises(RuntimeError, match="compile failed"):
        with preserve_final_options():
            raise RuntimeError("compile failed")
    assert numerics.effective_options(strict) == ()


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
        path = tmp_path / "hex-aot" / (entry["cubin_sha256"] + ".cubin")
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
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import write
    primary = tmp_path / "kernel-pack-cuda13.json"
    sidecar = tmp_path / "kernel-pack-cuda12.json"
    identities = {n: {"nvrtc": [n, 9]} for n in (12, 13)}
    write(primary, identities[13], [row()])
    write(sidecar, identities[12], [row()])
    from woof.hex import assets
    pins = {'schema': assets.SCHEMA, 'release_assets': {}, 'assets': {}}
    for n, path in [(12, sidecar), (13, primary)]:
        pins['assets'][path.name] = {'kind': 'kernel-pack', 'select': {'nvrtc_major': n},
                                    'compiler': identities[n], 'bytes': path.stat().st_size,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(assets, 'load_pins', lambda: pins)
    monkeypatch.setattr(assets, 'checkout_asset', lambda name: tmp_path / name)
    monkeypatch.setattr(assets, 'cache_root', lambda cache=None: tmp_path / 'cache')
    monkeypatch.setenv('WOOF_HEX_ASSETS_FIRST_USE', 'off')
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    monkeypatch.delenv("WOOF_HEX_AOT_PACK", raising=False)
    if explicit == "environment":
        monkeypatch.setenv("WOOF_HEX_AOT_PACK", str(primary))
    selected = Pack(primary if explicit == "argument" else None, identity=identities[major])
    if explicit:
        assert selected.reason == "pack_invalid_or_compiler_mismatch"
    else:
        assert selected.reason is None
        assert selected.compiler == identities[major]
    # A filename cannot admit a different minor compiler or corrupt envelope.
    if major == 12 and not explicit:
        assert Pack(identity={"nvrtc": [12, 8]}).reason == "pack_missing"

        primary.unlink()
        assert Pack(identity=identities[12]).reason is None
        sidecar.write_text('{"schema": "corrupt"}')
        assert Pack(identity=identities[12]).reason == "pack_missing"


def test_build_inventory_contains_production_and_both_architectures():
    from woof.hex.cuda_backend.aot_build import inventory
    rows = inventory()
    modules = {r["module_key"] for r in rows}
    assert {"hexcore.cuda_regional_v841", "hexcore.cuda_driver", "hexcore.cuda_physics_prep_v841"} <= modules
    # The shape forwarder contains a runtime raw_kernel call. Discovery must
    # ignore that ABI while retaining every caller's generic carrier.
    assert {f"hexcore.cuda_{family}_v841" for family in
            ("acoustic", "dynamics", "transport", "horizontal")} <= modules
    assert "woof.hex.cuda_shape" not in modules
    assert all(r["source_sha256"] == source_hash(r["source"]) for r in rows)


def test_direct_physics_pack_serves_bytes_and_corruption_uses_exact_fallback(tmp_path, monkeypatch):
    """A cold direct physics loader must use the pack, and never load corrupt bytes."""
    import sys
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import write

    entry = {**row(), "module_key": "nvrtc.direct." + source_hash("source"),
             "compile_request": aot.direct_metadata("physics.cu")}
    path = tmp_path / "direct.json"
    write(path, {"compiler": "test"}, [entry])
    selected = Pack(path, identity={"compiler": "test"})
    monkeypatch.setattr(aot, "Pack", lambda: selected)
    calls = []
    def compile_(source, **kwargs):
        calls.append((source, kwargs))
        return b"fallback", None
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(
        compiler=SimpleNamespace(compile_using_nvrtc=compile_)))
    assert aot.direct_image("source", ["--fmad=false"], "120", filename="physics.cu") == b"\x7fELFtest-image"
    assert not calls
    selected.rows[0]["cubin_sha256"] = "0" * 64
    assert aot.direct_image("source", ["--fmad=false"], "sm_120", filename="physics.cu") == b"fallback"
    assert calls == [("source", dict(options=("--fmad=false",), arch="120", filename="physics.cu"))]


def test_direct_physics_inventory_covers_both_surface_variants_and_numerics():
    """Wheel discovery must include the raw NVRTC sites omitted by the old AST scan."""
    from woof.hex.cuda_backend.aot_build import physics_inventory
    for mode in ("strict", "fast"):
        rows = physics_inventory(mode)
        families = [row["physics_family"] for row in rows]
        # Fast numerics also runs TEMPO as the phase modules P and S (tempo_phase_v303); strict never compiles them.
        fast_only = {"tempo_rrfs_phase", "tempo_rrfs_tail"} if mode == "fast" else set()
        # TEMPO's state reflectivity (tempo_refl_v303) compiles its strict text under both numerics: declared once.
        strict_only = {"tempo_rrfs_refl"} if mode == "strict" else set()
        assert set(families) == {"bl_mynn_adapter", "bl_mynn", "bl_mynnedmf", "sf_mynn", "sfclay_mm5",
                                 "rrtmg_lw", "rrtmg_sw", "noah_shared_tables", "gf_mpas", "thompson", "thompson_workspace", "thompson_aero_workspace", "tempo_rrfs",
                                 "rrfs_sfclay_mm5", "noah_seaice", "rrfs_noah_seaice",
                                 "wsm6", "wsm6_workspace", "rrfs_wsm6_radar", "rrfs_wsm6_relhum", "ntiedtke", "ysu", "ysu_gwdo", "ugwp_ngw",
                                 "rrfs_smoke", "nssl_rrfs", *fast_only, *strict_only}
        assert families.count("sf_mynn") == families.count("sfclay_mm5") == 2
        # MYNN: exact text and every correct-mode text set (RP7, the MPAS-route default).
        from woof.hex.mpasphys.bl_mynn_v842 import MYNN_TEXT_CORRECTIONS
        assert families.count("bl_mynn") == 2 ** len(MYNN_TEXT_CORRECTIONS) >= 2
        assert all(families.count(name) == 1 for name in
                   ("rrfs_sfclay_mm5", "noah_seaice", "rrfs_noah_seaice", "rrfs_wsm6_relhum"))
        assert all(families.count(name) == 2 for name in ("wsm6_workspace", "rrfs_wsm6_radar", "thompson"))
        # WSM6: the exact text and the correct-mode text (WB3, WB5), at 55 and 59 levels.
        assert families.count("wsm6") == 4
        assert all(row["compiler_route"] == ("raw-nvrtc" if row["physics_family"] in {"gf_mpas", "rrfs_smoke", "nssl_rrfs"}
                                               else "direct-nvrtc") for row in rows)
        expected = "--fmad=false" if mode == "strict" else "--fmad=true"
        assert all(expected in row["options"] for row in rows)
        noah = next(row for row in rows if row["physics_family"] == "noah_shared_tables")
        assert "--ftz=false" in noah["options"]
        from woof.hex.mpasphys import ugwp_ngw_v842 as ngw
        row = next(row for row in rows if row["physics_family"] == "ugwp_ngw")
        assert row["source"] == ngw.cuda_source(59)
        assert row["compile_request"]["filename"] == "ugwp_ngw_v842.cu"
        for entry in ("ngw_initialize", "ngw_run", "ngw_writeback"):
            assert f'void {entry}(' in row["source"]
        # YSU's live compile_using_nvrtc call uses an in-memory program name.
        ysu = next(row for row in rows if row["physics_family"] == "ysu")
        assert ysu["compile_request"] == {
            "route": "direct-nvrtc", "filename": "kern.cu", "cache_in_memory": True}
        assert all(row["compile_request"] == {
            "route": "direct-nvrtc", "filename": "ysu_gwdo_v842.cu", "cache_in_memory": True}
            for row in rows if row["physics_family"] == "ysu_gwdo")


def test_incremental_build_preserves_other_images_and_compiles_only_missing(tmp_path, monkeypatch):
    from concurrent import futures
    from woof.hex.cuda_backend import aot_build
    from woof.hex.cuda_backend.aot import ARCHITECTURES
    from woof.hex.kernel_pack import read, write
    from woof.certify import compile_platform

    identity = {"compiler": "test"}
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    monkeypatch.setattr(aot_build, "compiler_identity", lambda: identity)
    monkeypatch.setattr(compile_platform, "nvrtc_banner", lambda: "test banner")
    monkeypatch.setattr(compile_platform, "nvrtc_build", lambda _: "test")
    monkeypatch.setattr(compile_platform, "nvrtc_build_id", lambda _: "CL-test")
    class InlinePool:
        def __init__(self, workers, **kwargs):
            assert workers == 2
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, fn, jobs, **kwargs):
            return map(fn, jobs)
    monkeypatch.setattr(futures, "ProcessPoolExecutor", InlinePool)
    calls = []
    def compile_(entry, arch):
        calls.append(arch)
        return row(arch)
    monkeypatch.setattr(aot_build, "compile_entry", compile_)
    other = {**row(), "module_key": "unrelated.physics"}
    retained = [row(ARCHITECTURES[0]), other]
    path = tmp_path / "shared-pack.json"
    write(path, identity, retained)
    result = aot_build.build(path, base_pack=path, entries=[row()], workers=2)
    assert calls == list(ARCHITECTURES[1:])
    assert result["compiled"] == 3 and result["reused"] == 1
    assert read(path, identity)[:2] == retained
    assert result["entries"] == 5


def test_incremental_build_refuses_wrong_compiler_and_corrupt_reuse(tmp_path, monkeypatch):
    from woof.hex.cuda_backend import aot_build
    from woof.hex.kernel_pack import write
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    monkeypatch.setattr(aot_build, "compiler_identity", lambda: {"compiler": "test"})
    path = tmp_path / "pack.json"
    write(path, {"compiler": "other"}, [row()])
    with pytest.raises(ValueError, match="base pack cannot be reused"):
        aot_build.build(path, base_pack=path, entries=[row()])
    write(path, {"compiler": "test"}, [{**row(), "cubin_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="corrupt image"):
        aot_build.build(path, base_pack=path, entries=[row()])


def test_raw_physics_pack_and_fallback_keep_exact_nvrtc_options(tmp_path, monkeypatch):
    import sys
    from woof.hex.cuda_backend import aot
    from woof.hex import kernel_warm
    # The process-wide warm store (kernel_warm._STATE) serves raw NVRTC images it already holds; an earlier
    # test that ran real CuPy with WOOF_HEX_PREP_CACHE set installs it, and this test then never reaches
    # its own cache or pack logic (G0 order: tests/test_contract_hostprep.py first). Measure without it.
    monkeypatch.setattr(kernel_warm, "_STATE", None)
    monkeypatch.setattr(kernel_warm, "install", lambda **_: None)
    from woof.hex.kernel_pack import write
    entry = {**row(), "module_key": "nvrtc.raw." + source_hash("source"),
             "compile_request": {"route": "raw-nvrtc", "filename": "gf.cu"}}
    target = tmp_path / 'pack.json'
    write(target, {'compiler': 'test'}, [entry])
    selected = Pack(target, identity={'compiler': 'test'})
    monkeypatch.setattr(aot, 'Pack', lambda: selected)
    calls = []
    nvrtc = SimpleNamespace(
        createProgram=lambda *args: calls.append(('create', args)) or 1,
        compileProgram=lambda *args: calls.append(('compile', args)),
        getCUBIN=lambda program: b'fallback',
        destroyProgram=lambda program: calls.append(('destroy', program)))
    monkeypatch.setitem(sys.modules, 'cupy_backends.cuda.libs', SimpleNamespace(nvrtc=nvrtc))
    assert aot.raw_image('source', ['--fmad=false'], '120', filename='gf.cu') == b'\x7fELFtest-image'
    assert not calls
    selected.rows[0]['cubin_sha256'] = '0' * 64
    assert aot.raw_image('source', ['--fmad=false'], '120', filename='gf.cu') == b'fallback'
    assert calls == [('create', ('source', 'gf.cu', (), ())),
                     ('compile', (1, ('--fmad=false', '--gpu-architecture=sm_120'))),
                     ('destroy', 1)]


def test_c1_mark_records_named_fallback(monkeypatch):
    from woof.hex.cuda_backend.aot import mark
    from woof.hex import time_ledger
    events = []
    monkeypatch.setattr(time_ledger, "emit", lambda *a, **kw: events.append((a, kw)))
    mark("jit", "hexcore.test", 0.125, reason="source_hash_mismatch")
    assert events == [(("forecast.construct",), {"attrs": {"mark": "kernel.load", "source": "jit", "module_key": "hexcore.test", "count": 1, "seconds": 0.125, "reason": "source_hash_mismatch"}})]


@pytest.mark.parametrize("flag", ["jitify", "to_ltoir", "enable_cooperative_groups"])
def test_recorded_routes_needing_linking_are_refused(flag):
    from woof.hex.cuda_backend.aot import recipe
    with pytest.raises(ValueError, match="requires linking"):
        recipe({"source": "source", flag: True})


def test_recorded_request_keeps_template_and_extra_source_identity():
    from woof.hex.cuda_backend.aot import recipe
    entry = recipe({"source": "source", "extra_source": "headers",
                    "name_expressions": ["copy<float>"], "options": ["--fmad=false"]})
    assert entry["compile_request"]["extra_source"] == "sha256:" + source_hash("headers")
    assert entry["compile_request"]["name_expressions"] == ["copy<float>"]
    assert entry["options"] == ["--fmad=false"]


def test_same_source_options_select_the_requested_template_variant(tmp_path, monkeypatch):
    first, second = row(), row()
    first["compile_request"] = {"name_expressions": ["copy<float>"]}
    second["compile_request"] = {"name_expressions": ["copy<double>"]}
    selected = pack(tmp_path, monkeypatch, [first, second])
    chosen, reason = selected.select("hexcore.test", "source", ["--fmad=false"], "sm_120",
                                     metadata=second["compile_request"])
    assert chosen == second and reason is None


def test_cached_banner_checks_library_bytes_not_just_version(monkeypatch):
    from woof.certify import compile_platform
    from woof.hex.cuda_backend.aot import cache_compiler_banner
    banner = "// Cuda compilation tools, release 13.3, V13.3.0\n// Compiler Build ID: CL-12345"
    current = {"sha": "a" * 64}
    probes = []
    monkeypatch.setattr(compile_platform, "nvrtc_banner", lambda: probes.append(True) or "probe")
    monkeypatch.setattr(compile_platform, "nvrtc_library", lambda: {"status": "resolved", "sha256": current["sha"]})
    selected = SimpleNamespace(rows=[{"nvrtc_banner": banner}],
                               compiler={"nvrtc": [13, 3], "nvrtc_library_sha256": current["sha"]})
    cache_compiler_banner(selected)
    assert compile_platform.nvrtc_banner() == banner and probes == []
    current["sha"] = "b" * 64
    assert compile_platform.nvrtc_banner() == "probe" and probes == [True]


def test_regional_source_guard_covers_the_new_module_loader():
    from woof.hex.cuda_backend.regional_admission import REGIONAL_KERNEL_SOURCES
    assert "src/hexcore/cuda_backend/aot.py" in REGIONAL_KERNEL_SOURCES


def test_options_are_portable_across_install_roots():
    from woof.hex.cuda_backend.aot import local_options, portable_options
    first = ["-I/venv-a/lib/python3.12/site-packages/cupy/_core/include", "--std=c++17",
             "-I/venv-a/lib/python3.12/site-packages/nvidia/cu13/include", "-I/usr/local/cuda/include"]
    second = [option.replace("/venv-a/", "/opt/other/") for option in first]
    a = portable_options(first, roots=("/venv-a/lib/python3.12/site-packages",))
    b = portable_options(second, roots=("/opt/other/lib/python3.12/site-packages",))
    assert a == b and a[0] == "-I@SITE@/cupy/_core/include" and a[3] == "-I/usr/local/cuda/include"
    assert local_options(a, roots=("/venv-a/lib/python3.12/site-packages",)) == first


def test_compiler_mismatch_names_the_pack_compiler_and_the_remedy(tmp_path, monkeypatch, capsys):
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import write
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    monkeypatch.setattr(aot, "_WARNED", set())
    path = tmp_path / "pack.json"
    write(path, {"cupy": "14.2.0", "cupy_package": "cupy-cuda13x==14.2.0",
                 "nvrtc_package": "nvidia-cuda-nvrtc==13.4.92"}, [row()])
    selected = Pack(path, identity={"cupy": "14.3.0", "cupy_package": "cupy-cuda13x==14.3.0",
                                    "nvrtc_package": "nvidia-cuda-nvrtc==13.5.1"})
    assert selected.reason == "pack_invalid_or_compiler_mismatch"
    error = capsys.readouterr().err
    assert "pip install cupy-cuda13x==14.2.0 nvidia-cuda-nvrtc==13.4.92" in error
    assert "cupy-cuda13x==14.3.0" in error


def test_rows_without_source_text_select_by_digest(tmp_path, monkeypatch):
    shipped = row()
    del shipped["source"]
    selected = pack(tmp_path, monkeypatch, [shipped])
    assert selected.select("hexcore.test", "source", ["--fmad=false"], "sm_120")[1] is None
    assert selected.select("hexcore.test", "changed", ["--fmad=false"], "sm_120") == (None, "source_hash_mismatch")


def test_direct_nvrtc_requests_become_their_own_rows():
    from woof.hex.cuda_backend.aot import DIRECT_ROUTE, recipe
    entry = recipe({"source": "radiation", "options": ["-std=c++17", "--ftz=false"],
                    "route": DIRECT_ROUTE, "filename": "rrtmg.cu"})
    assert entry["module_key"] == "nvrtc.direct." + source_hash("radiation")
    assert entry["compiler_route"] == DIRECT_ROUTE
    assert entry["compile_request"] == {"route": DIRECT_ROUTE, "filename": "rrtmg.cu"}
    assert entry["options"] == ["-std=c++17", "--ftz=false"]


def test_pack_image_refuses_a_cubin_that_fails_its_digest():
    shipped = row()
    assert Pack.image(shipped) == b"\x7fELFtest-image"
    assert Pack.image({**shipped, "cubin_sha256": "0" * 64}) is None


@pytest.mark.parametrize("family", [12, 13])
def test_shipped_pack_covers_every_declared_module_on_every_architecture(family):
    """THE BREAKAGE: a kernel edited without rebuilding the pack JITs on every fresh box.

    Checked per numerics mode with the exact options the pack selects on: a pack holding only
    strict rows misses every hex kernel of a default (fast) run (owner ruling 2026-10-16).
    Rebuild with: python tools/build_hex_aot.py --output release-assets/kernel-packs/<name>
    (in a venv holding the pack's CuPy and NVRTC, named in its "compiler" field).
    """
    from woof.hex.cuda_backend.aot import ARCHITECTURES
    from woof.hex.cuda_backend.aot_build import inventory
    from woof.hex import assets
    name = f"kernel-pack-cuda{family}.json"
    pin = assets.load_pins()['assets'][name]
    path = assets.checkout_asset(name)
    if path is None:
        path = assets.asset_path(name, pin)
    if not path.is_file():
        pytest.skip("optional release-asset inventory needs staged or private checkout packs")
    assets.verify(path, pin)
    document = json.loads(path.read_text())
    held = {(e["arguments"]["module_key"], e["arguments"]["source_sha256"],
             tuple(e["arguments"]["options"]), e["arguments"]["arch"]) for e in document["entries"]}
    for mode in ("fast", "strict"):
        missing = sorted({row["module_key"] for row in inventory(numerics=mode, compiler_family=family) for arch in ARCHITECTURES
                          if (row["module_key"], row["source_sha256"], tuple(row["options"]), arch) not in held})
        assert not missing, f"kernel pack is stale for {mode} numerics: {missing}"


@pytest.mark.parametrize("catalog", ["compiler-recipes.json", "compiler-recipes-cuda12.json"])
def test_recorded_direct_units_are_held_under_both_numerics(catalog):
    """THE BREAKAGE: catalog rows recorded from fast runs only gave the shipped pack no strict image
    of RRTMG SW, YSU GWDO, the cloud fraction, sea-ice and surface units, so a fresh box's first
    strict step compiled them on one core (woof hex 0.4.9, VRFS 2026-10-28: 547 s on one card,
    one module 519 s).  Record each route under both numerics, then
    tools/record_hex_aot_recipes.py --merge (CUDA 12 catalog: --routes direct-nvrtc)."""
    from pathlib import Path
    import woof.hex
    from woof.hex.cuda_backend.aot_build import catalog_numerics_gaps
    rows = json.loads((Path(woof.hex.__file__).parent / "data" / catalog).read_text())
    assert not catalog_numerics_gaps(rows)


def test_numerics_gaps_pair_units_by_file_and_entry_point():
    from woof.hex.cuda_backend.aot_build import numerics_gaps
    def direct(source, options, filename="kern.cu"):
        return {"route": "direct-nvrtc", "filename": filename, "source": source, "options": options}
    fast = ["--fmad=true", "-DHEX_NUMERICS_FAST=1"]
    strict = ["--fmad=false", "--ftz=false"]
    sw = 'extern "C" __global__ void sw_a(float*x){}\n__global__ void __launch_bounds__(64) sw_b(){}'
    gw = 'extern "C" __global__ void gwdo(float*x){}'
    assert numerics_gaps([direct(sw, fast), direct(gw, fast), direct(gw + " ", strict)]) == [
        ("strict", "kern.cu", ("sw_a", "sw_b"))]
    # A strict unit needs a fast or flag-free (engine) row; one shared entry point pairs.
    assert numerics_gaps([direct(sw, strict), direct('__global__ void sw_a(){}', [])]) == []
    assert numerics_gaps([direct(sw, strict)]) == [("fast", "kern.cu", ("sw_a", "sw_b"))]
    # Same entry points in another file, or a CuPy-route row, do not pair.
    assert [g[:2] for g in numerics_gaps([direct(gw, fast), direct(gw, strict, "other.cu")])] == [
        ("fast", "other.cu"), ("strict", "kern.cu")]
    assert numerics_gaps([{"source": gw, "options": fast}]) == []
    # A declared fast-only unit (strict never launches it) needs no strict row; only units inside it are exempt.
    unit = ("kern.cu", frozenset({"sw_a", "sw_b"}))
    assert numerics_gaps([direct(sw, fast)], fast_only={unit}) == []
    assert numerics_gaps([direct(sw, fast), direct(gw, fast)], fast_only={unit}) == [
        ("strict", "kern.cu", ("gwdo",))]


def test_inventory_keeps_compiler_family_options_exact(tmp_path):
    from woof.hex.cuda_backend.aot_build import inventory
    data = tmp_path / "data"
    data.mkdir()
    source = 'extern "C" __global__ void family_catalog_probe(){}'
    for family, include in [(12, "cuda_runtime"), (13, "cu13")]:
        (data / f"compiler-recipes-cuda{family}.json").write_text(json.dumps([
            dict(source=source, options=[f"-I@SITE@/nvidia/{include}/include"])]))
    rows = [r for r in inventory(tmp_path) if r["source_sha256"] == source_hash(source)]
    assert {tuple(r["options"]) for r in rows} == {
        ("-I@SITE@/nvidia/cuda_runtime/include",), ("-I@SITE@/nvidia/cu13/include",)}

    for family, include in [(12, "cuda_runtime"), (13, "cu13")]:
        rows = [r for r in inventory(tmp_path, compiler_family=family)
                if r["source_sha256"] == source_hash(source)]
        assert [r["options"] for r in rows] == [[f"-I@SITE@/nvidia/{include}/include"]]
    with pytest.raises(ValueError, match="no compiler recipe catalog for CUDA 14"):
        inventory(tmp_path, compiler_family=14)


def test_catalog_recipe_binds_a_system_toolkit_include(monkeypatch):
    """A CUDA 13.0 system toolkit shadowing the pip NVRTC: CuPy compiles with
    -I/usr/local/cuda/include, so rows formed for this compiler must carry it."""
    from woof.hex.cuda_backend import aot_build
    entry = dict(source='extern "C" __global__ void toolkit_probe(){}',
                 options=["-I@SITE@/cupy/_core/include", "-I@SITE@/nvidia/cu13/include"])
    compiler = {"nvrtc": [13, 0]}
    monkeypatch.setattr(aot_build, "toolkit_includes", lambda: ["-I/usr/local/cuda/include"])
    assert aot_build.catalog_recipe(entry, compiler)["options"] == [
        "-I@SITE@/cupy/_core/include", "-I/usr/local/cuda/include"]
    # Without a compiler (inspection, the pack staleness check) the recorded layout stands.
    assert aot_build.catalog_recipe(entry)["options"] == entry["options"]
    monkeypatch.setattr(aot_build, "toolkit_includes", lambda: None)
    assert aot_build.catalog_recipe(entry, compiler)["options"] == entry["options"]


def test_cached_program_filename_binds_installed_headers(monkeypatch):
    from woof.hex import kernel_plan
    from woof.hex.cuda_backend.aot import recipe
    from woof.hex.cuda_backend.aot_build import cached_program_filename
    captured = []
    monkeypatch.setattr(kernel_plan, "_cupy_extra_sources", lambda: (None, "headers"))
    def entry(arguments, target, extra):
        captured.append((arguments, target, extra))
        return "runtime-key.cubin"
    monkeypatch.setattr(kernel_plan, "_cupy_entry", entry)
    recorded = recipe({"source": "source", "extra_source": "headers",
                       "name_expressions": ["copy<float>"]})
    assert cached_program_filename(recorded, ("-ftz=true",), "120") == "runtime-key.cubin.cu"
    assert captured == [({"source": "source", "options": ("-ftz=true",),
                         "name_expressions": ["copy<float>"]}, "120", "headers")]
    recorded["compile_request"]["extra_source"] = "sha256:" + source_hash("other headers")
    with pytest.raises(ValueError, match="headers differ"):
        cached_program_filename(recorded, (), "120")
    assert len(captured) == 1


@pytest.mark.parametrize("mode,expected", [("fast", ("-std=c++17",)),
                                            ("strict", ("-std=c++17", "--fmad=false"))])
def test_cached_bridge_applies_runtime_numerics_before_pack_selection(monkeypatch, mode, expected):
    import sys
    from woof import numerics
    from woof.hex.cuda_backend import aot
    monkeypatch.setattr(numerics, "_mode", mode)
    monkeypatch.setattr(aot, "cache_compiler_banner", lambda selected: None)
    options = []
    result = object()
    def load(cp, directory, key, source, flags, sm, **kwargs):
        options.append(flags)
        return SimpleNamespace(module=result), None, None
    selected = SimpleNamespace(rows=[{"compile_request": {}}], load=load)
    compiler = SimpleNamespace(_compile_with_cache_cuda=lambda *a, **k: pytest.fail("unexpected JIT"),
                               compile_using_nvrtc=lambda *a, **k: pytest.fail("unexpected direct JIT"))
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(compute_capability="120")))
    aot._bridge(cp, selected)
    assert compiler._compile_with_cache_cuda("source", ("-std=c++17", "--fmad=false")) is result
    assert options == [expected]
@pytest.fixture
def family_packs(tmp_path, monkeypatch):
    from woof.hex import assets
    from woof.hex.kernel_pack import write
    pins = {'schema': assets.SCHEMA, 'release_assets': {}, 'assets': {}}
    paths = {}
    for major in (12, 13):
        identity = {'nvrtc': [major, 4], 'nvrtc_library_sha256': str(major) * 32}
        path = tmp_path / f'kernel-pack-cuda{major}.json'
        write(path, identity, [row()])
        paths[major] = path
        pins['assets'][path.name] = {'kind': 'kernel-pack', 'select': {'nvrtc_major': major},
                                    'compiler': identity, 'bytes': path.stat().st_size,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(assets, 'load_pins', lambda: pins)
    monkeypatch.setattr(assets, 'checkout_asset', lambda name: tmp_path / name)
    monkeypatch.setattr(assets, 'cache_root', lambda cache=None: tmp_path / 'cache')
    monkeypatch.delenv('WOOF_HEX_AOT', raising=False)
    monkeypatch.delenv('WOOF_HEX_AOT_PACK', raising=False)
    return paths


@pytest.mark.parametrize("major", [12, 13])
def test_installed_toolchain_selects_matching_pinned_pack(monkeypatch, major, family_packs):
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import compiler_of
    identity = compiler_of(family_packs[major])
    monkeypatch.setattr(aot, 'compiler_identity', lambda: identity)
    selected = aot.Pack()
    assert selected.reason is None and len(selected.rows) > 0
    assert selected.compiler == identity
    assert aot.Pack(family_packs[25-major]).reason == 'pack_invalid_or_compiler_mismatch'


def test_pack_override_and_disabled_mode_preserve_refusals(monkeypatch, family_packs):
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import compiler_of
    identity = compiler_of(family_packs[12])
    monkeypatch.setattr(aot, 'compiler_identity', lambda: identity)
    monkeypatch.setenv('WOOF_HEX_AOT_PACK', str(family_packs[13]))
    assert aot.Pack().reason == 'pack_invalid_or_compiler_mismatch'
    assert aot.Pack(family_packs[12]).reason is None
    monkeypatch.setenv('WOOF_HEX_AOT', 'off')
    assert aot.Pack().reason == 'aot_disabled'


def test_selected_family_still_requires_exact_nvrtc_library(monkeypatch, family_packs):
    from woof.hex.cuda_backend import aot
    from woof.hex.kernel_pack import compiler_of
    identity = {**compiler_of(family_packs[12]), 'nvrtc_library_sha256': '0' * 64}
    monkeypatch.setattr(aot, '_WARNED', set())
    assert aot.Pack(identity=identity).reason == 'pack_missing'
    # An explicit path still goes through the codec's exact compiler check.
    assert aot.Pack(family_packs[12], identity=identity).reason == 'pack_invalid_or_compiler_mismatch'



@pytest.mark.parametrize("route", ["helper", "bridge"])
@pytest.mark.parametrize("exact", [False, True])
def test_direct_pack_selection_uses_final_wrf_options(tmp_path, monkeypatch, route, exact):
    """An outer pack lookup must not bypass the engine's corrected strict key."""
    import sys
    from woof import numerics
    from woof.hex.cuda_backend import aot

    monkeypatch.setattr(numerics, "_mode", "fast")
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1" if exact else "0")
    options = ("--std=c++17",)
    fast = row()
    fast.update(module_key="nvrtc.direct." + source_hash("source"), options=list(options),
                compile_request=aot.direct_metadata("named.cu"), name_mapping={"k": "fast"})
    from woof.wrf_exact import effective_options as wrf_options
    strict = dict(fast, options=list(wrf_options(options)))
    strict_blob = b"\x7fELFstrict-image"
    strict.update(cubin_base64=base64.b64encode(strict_blob).decode(),
                  cubin_sha256=hashlib.sha256(strict_blob).hexdigest(), name_mapping={"k": "strict"})
    selected = pack(tmp_path, monkeypatch, [fast, strict] if exact else [fast])
    monkeypatch.setattr(aot, "Pack", lambda: selected)
    monkeypatch.setattr(aot, "cache_compiler_banner", lambda selected: None)
    monkeypatch.setattr(aot, "mark", lambda *args, **kwargs: None)
    calls = []
    def compile_(*args, **kwargs):
        calls.append((args, kwargs))
        return b"fallback", None
    compiler = SimpleNamespace(_compile_with_cache_cuda=compile_, compile_using_nvrtc=compile_)
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    if route == "bridge":
        aot._bridge(None, selected)
        image, mapping = compiler.compile_using_nvrtc("source", options, "120", "named.cu")
        assert mapping == {"k": "strict" if exact else "fast"}
    else:
        image = aot.direct_image("source", options, "120", filename="named.cu")
    assert image == (strict_blob if exact else b"\x7fELFtest-image")
    assert calls == []


def test_direct_recorder_and_plan_coverage_use_final_wrf_options(monkeypatch):
    import sys
    from woof import numerics
    from woof.hex import kernel_plan
    from woof.hex.cuda_backend import aot

    monkeypatch.setattr(numerics, "_mode", "fast")
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    monkeypatch.setenv("WOOF_HEX_AOT_RECORD", "unused-single-file.json")
    monkeypatch.setattr(aot, "_RECORDED", {})
    monkeypatch.setattr(__import__("atexit"), "register", lambda function: None)
    compiler = SimpleNamespace(_compile_with_cache_cuda=lambda *a, **k: None,
                               compile_using_nvrtc=lambda *a, **k: (b"image", None))
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    aot._install_recorder()
    options = ("--std=c++17",)
    compiler.compile_using_nvrtc("source", options, "120", "named.cu")
    recorded = next(iter(aot._RECORDED.values()))
    from woof.wrf_exact import effective_options as wrf_options
    assert tuple(recorded["options"]) == wrf_options(options)
    entry = dict(source_sha256=source_hash("source"), options=recorded["options"],
                 arch="sm_120", compiler_route="direct-nvrtc")
    held = kernel_plan._aot_coverage(SimpleNamespace(reason=None, rows=[entry]))
    arguments = dict(source="source", options=list(options), arch="device",
                     name_expressions=None, method=None, filename="named.cu")
    assert kernel_plan._aot_cached(arguments, "120", held)


def test_direct_memory_image_requires_exact_program_name_mode(tmp_path, monkeypatch):
    """Memory-only direct compiles use an empty NVRTC name, never a disk image."""
    import sys
    from woof.hex.cuda_backend import aot
    options = ('--std=c++17',)
    disk = row()
    disk.update(module_key='nvrtc.direct.' + source_hash('source'), options=list(options),
                compile_request=aot.direct_metadata('named.cu'), name_mapping=None)
    selected = pack(tmp_path, monkeypatch, [disk])
    monkeypatch.setattr(aot, 'cache_compiler_banner', lambda selected: None)
    monkeypatch.setattr(aot, 'mark', lambda *args, **kwargs: None)
    monkeypatch.setattr(aot, 'effective_options', tuple)
    calls = []
    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return b'fallback', None
    compiler = SimpleNamespace(_compile_with_cache_cuda=fallback, compile_using_nvrtc=fallback)
    monkeypatch.setitem(sys.modules, 'cupy.cuda', SimpleNamespace(compiler=compiler))
    aot._bridge(None, selected)
    args = ('source', options, '120', 'named.cu')
    assert compiler.compile_using_nvrtc(*args, cache_in_memory=True) == (b'fallback', None)
    assert calls == [(args, {'cache_in_memory': True})]
    # Supplying the exact memory entry serves bytes without writing any cache file.
    memory = dict(disk, compile_request=aot.direct_metadata('named.cu', True))
    selected.rows.append(memory)
    selected._by_key[disk['module_key']].append(memory)
    before = {p.name for p in tmp_path.iterdir()}
    assert compiler.compile_using_nvrtc(*args, cache_in_memory=True) == (b'\x7fELFtest-image', None)
    assert compiler.compile_using_nvrtc(*args, cache_in_memory=False) == (b'\x7fELFtest-image', None)
    assert len(calls) == 1 and {p.name for p in tmp_path.iterdir()} == before
    built = aot.recipe(dict(source='source', options=options, **aot.direct_metadata('named.cu', True)))
    assert built['compile_request'] == memory['compile_request']


def test_nvrtc_13_2_unknown_build_id_is_pinned_by_its_library():
    """System NVRTC 13.2.86 prints 'Compiler Build ID: UNKNOWN'; the library digest pins it."""
    from woof.hex.cuda_backend.compile_contract import build_id_accepted
    assert build_id_accepted("CL-36424714") and build_id_accepted("UNKNOWN")
    assert not build_id_accepted("") and not build_id_accepted("unavailable") and not build_id_accepted(None)


def _worker_gone(process, timeout=10.0):
    """True once ``process`` has exited, judged so that another reaper cannot fool it.

    The pool's own manager thread also joins its workers once one is terminated (the pool turns broken).
    When it reaps first, this thread's waitpid gets ECHILD, multiprocessing reads that as "still running"
    and is_alive() stays True until the manager thread sets the return code -- the repr in the assertion
    still said "started".  Breakage this ends (public battery 3.11, 4 of 6 runs on 2026-10-26, both exit
    tests at once): join(10) returns at once on a ready sentinel, so nothing ever yielded to that thread.
    Sleeping yields; a reaped or zombie pid is also an exit.
    """
    import os
    import time
    process.join(timeout)
    deadline = time.monotonic() + timeout
    while process.exitcode is None and time.monotonic() < deadline:
        try:
            state = open(f"/proc/{process.pid}/stat", "rb").read().rsplit(b")", 1)[1].split()[0]
        except FileNotFoundError:
            return True        # reaped by the pool's own thread
        except OSError:
            state = None       # no /proc (not Linux): wait for the exit code
        if state in (b"Z", b"X"):
            return True        # exited, waiting to be reaped
        time.sleep(0.05)
    return process.exitcode is not None


def test_unrequested_pool_compiles_do_not_hold_the_exit():
    """A finished forecast waited 70-90 s for one worker compiling a module nobody asked for."""
    import concurrent.futures
    import multiprocessing
    import time
    from woof.hex import kernel_plan
    pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    processes = pool._processes
    pool.submit(time.sleep, 120)
    deadline = time.monotonic() + 60
    while not processes and time.monotonic() < deadline:
        time.sleep(0.05)
    pool.shutdown(wait=False)  # what kernel_plan.close() does; it drops the attribute
    started = time.monotonic()
    kernel_plan._drop_at_exit(pool, processes)
    for process in list(processes.values()):
        assert _worker_gone(process)
    assert time.monotonic() - started < 10


def test_a_failing_shutdown_still_terminates_an_idle_worker():
    """The hl-3 headline run (2026-10-24): the forecast wrote its receipt, then waited 16 min joining one
    idle spawn compile worker. The exit drop terminates the workers even when shutdown raises."""
    import concurrent.futures
    import multiprocessing
    import time
    from woof.hex import kernel_plan
    pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    processes = pool._processes
    pool.submit(time.sleep, 0).result(60)   # the worker is up and idle

    class Raising:
        def shutdown(self, **_kwargs):
            raise OSError("wakeup pipe closed")

    kernel_plan._drop_at_exit(Raising(), processes)
    for process in list(processes.values()):
        assert _worker_gone(process)
    pool.shutdown(wait=True, cancel_futures=True)


#: Run in a spawn compile worker by the builtin ``exec``: write a result header that promises 1 MiB, send
#: 1 KiB of it, then wait to be stopped -- a worker caught by the exit drop in the middle of sending its image.
_HALF_SENT_RESULT = """
import os, struct, sys, time
frame = sys._getframe(0)
while frame is not None and "result_queue" not in frame.f_locals:
    frame = frame.f_back
os.write(frame.f_locals["result_queue"]._writer.fileno(), struct.pack("!i", 1 << 20) + bytes(1024))
time.sleep(600)
"""


@pytest.mark.skipif(os.name != "posix", reason="the half-sent result is written with os.write on the POSIX result "
                    "pipe; on Windows that pipe is a named-pipe handle with no file descriptor")
def test_a_worker_stopped_mid_result_does_not_hold_the_exit():
    """RRFS restart proofs (an RTX 5090 host, 2026-10-29): finished forecasts never exited and held their card lease. A
    compile worker the exit drop stopped while it was sending its image left the pool's manager thread reading
    the rest forever, and concurrent.futures' exit hook joins that thread with no timeout."""
    import concurrent.futures
    import multiprocessing
    import time
    from woof.hex import kernel_plan
    pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    processes, results = pool._processes, pool._result_queue
    pool.submit(exec, _HALF_SENT_RESULT)
    manager = pool._executor_manager_thread
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:   # until the manager is inside the read of the half-sent result
        try:
            if "pipe_read" in open(f"/proc/self/task/{manager.native_id}/wchan").read():
                break
        except OSError:
            time.sleep(3)
            break
        time.sleep(0.05)
    pool.shutdown(wait=False)  # what kernel_plan.close() does at the end of the forecast
    try:
        kernel_plan._drop_at_exit(pool, processes, results)
        manager.join(20)
        assert not manager.is_alive()
    finally:
        results._writer.close()  # a red run must not also hang the test session's own exit
        manager.join(20)


#: A forecast process in miniature: it opens the compile pool from a thread that ends right after
#: submitting (hex-prewarm-setup does), checks its two compiling workers outlived that thread, prints
#: their pids, then ends without any exit hook (``exit``: os._exit(0)) or waits to be SIGKILLed (``kill``).
_POOL_OPENER = """
import os, sys, threading, time
from woof.hex import kernel_plan
pools = []
def setup():
    pool = kernel_plan.compiler_pool(2)
    pools.append(pool)
    for _ in range(2):
        pool.submit(time.sleep, 600)
thread = threading.Thread(target=setup)
thread.start()
thread.join()
pool = pools[0]
deadline = time.monotonic() + 60
while len(pool._processes) < 2 and time.monotonic() < deadline:
    time.sleep(0.05)
time.sleep(1.0)
print(" ".join(str(pid) for pid, process in pool._processes.items() if process.is_alive()), flush=True)
if sys.argv[1] == "exit":
    os._exit(0)
time.sleep(600)
"""


def _pid_running(pid):
    """A pid that exists and is not a zombie (a dead worker waiting for init to reap it is gone)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="worker liveness is read from /proc (Linux)")
@pytest.mark.parametrize("end", ["exit", "kill"])
def test_compile_workers_end_with_the_process_that_opened_them(end):
    """an RTX 5090 host, 2026-10-29 08:17-08:21Z: 89 spawn compile workers of forecasts that had ended (OOM-killed,
    or stopped with no exit hook) held 62.5 GB until killed by PID; the OOM killer then took release
    containers and lanes. The workers must also outlive the short-lived thread that submitted to them."""
    import signal
    import subprocess
    import sys
    import time
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    opener = subprocess.Popen([sys.executable, "-c", _POOL_OPENER, end], stdout=subprocess.PIPE, text=True,
                              env=environment)
    workers = []
    try:
        workers = [int(pid) for pid in opener.stdout.readline().split()]
        assert len(workers) == 2, "a compile worker died with the thread that submitted to it"
        if end == "kill":
            opener.send_signal(signal.SIGKILL)
        opener.wait(60)
        deadline = time.monotonic() + 10
        while any(_pid_running(pid) for pid in workers) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not [pid for pid in workers if _pid_running(pid)]
    finally:
        if opener.poll() is None:
            opener.kill()
            opener.wait(10)
        for pid in workers:
            if _pid_running(pid):
                os.kill(pid, signal.SIGKILL)   # a red run leaves no orphan behind


@pytest.mark.parametrize('mode', ['strict', 'fast'])
@pytest.mark.parametrize('corrections', [frozenset(), frozenset({'RP7'})])
def test_mynn_inventory_matches_the_forecast_source(mode, corrections, monkeypatch):
    import sys
    import types
    # MynnCudaModule only binds cupy when constructed (the source text is built on the host), so the CPU tier,
    # which installs no cupy, runs this with a stand-in module (HB2 gate HXCI red, 4 cases on 3.11 and 3.13).
    from woof.hex import numerics
    from woof.hex.cuda_backend.aot_build import physics_inventory
    from woof.hex.mpasphys import bl_mynn_v842 as bl, resources_build as resources
    monkeypatch.setattr(numerics, '_mode', mode)
    with monkeypatch.context() as stub:
        stub.setitem(sys.modules, 'cupy', sys.modules.get('cupy') or types.ModuleType('cupy'))
        module = bl.MynnCudaModule(shared_math_source=resources.mynn_maths(resources._vendored_maths()),
                                   corrections=corrections)
    rows = [r for r in physics_inventory(mode) if r.get('physics_family') == 'bl_mynn']
    expected = numerics.options(bl.MYNN_CUDA_OPTIONS, numerics=mode)
    assert any(r['source'] == module._source and tuple(r['options']) == expected for r in rows)
