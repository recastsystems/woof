"""The MPAS route's resources resolve from the install by portable key.

The v8.4.2 forecast uses the installed wheel's Registry and pool reader,
plus fetched DATA. Native referee decks resolve explicit site-built
resources by key. These checks run on the CPU with no network or compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from woof.hex.mpasphys.native_library_build import referee_builder

nlb = referee_builder()
from woof.hex.mpasphys import resources_build as rb

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_pool_reader_is_gpuwm_24b6f08_plus_the_field_parallel_transport():
    # woof 24b6f08c carries ecd4936f: zero-sized fork Registry dimensions allocate associated
    # empty arrays (MPAS-Model-rrfs mpas_field_routines.F:261-299); 0ec69a6 refused them.
    # hex fix/U-H3 then moved the transport onto a worker pool (same bytes per field), and
    # fix/rrfs-c3km-startup-split split one large field over threads (same bytes per field).
    source = ROOT / "rust/physics-pool/src/bin/rw_mpas_physics_pool.rs"
    assert _sha(source) == "1a42521cdcaa87bf8931b3ce5b5869bfe16f61c7519d9463903580a295886e83"
    cargo = (ROOT / "rust/physics-pool/Cargo.toml").read_text()
    assert 'name = "rw_mpas_physics_pool"' in cargo
    lock = (ROOT / "rust/physics-pool/Cargo.lock").read_text()
    for name, version in (("serde", "1.0.229"), ("serde_json", "1.0.151"), ("sha2", "0.10.9")):
        assert f'name = "{name}"\nversion = "{version}"' in lock


def test_the_source_manifest_pins_what_the_builders_pin():
    manifest = nlb.source_manifest()
    assert manifest["schema"] == "gpuwm-hex.mpas-source-manifest/v1"
    assert manifest["pins"] == {"mpas-model": rb.MPAS_MODEL_COMMIT, "mmm-physics": rb.MMM_PHYSICS_COMMIT,
                                "ugwp": rb.UGWP_COMMIT, "mpas-data": rb.MPAS_DATA_COMMIT}
    build_py = (ROOT / "tools/native_physics_reference/backend/build.py").read_text()
    assert f"PIN = '{rb.MPAS_MODEL_COMMIT}'" in build_py
    for origin in ("mmm-physics", "ugwp"):
        assert manifest["pins"][origin] in build_py
    # The Thompson-table generator and the library build read the same pinned bytes.
    seen = set()
    for name, (origin, path, sha) in rb.THOMPSON_SOURCES.items():
        for row in manifest["files"].values():
            if row["origin"] == origin and row["path"] == path:
                assert row["sha256"] == sha, name
                seen.add(name)
    assert {"module_mp_thompson.F", "mpas_atmphys_functions.F", "mp_radar.F90"} <= seen
    for name, (origin, path, sha) in rb.TABLE_FILES.items():
        assert manifest["tables"][name]["sha256"] == sha
    # Registry.xml and the files it includes are all present for the pool decoder.
    registry = [relative for relative in manifest["files"] if relative.startswith(rb.REGISTRY_PREFIXES)]
    assert "src/core_atmosphere/Registry.xml" in registry
    assert "src/core_atmosphere/physics/Registry_noahmp.xml" in registry
    assert "src/core_atmosphere/diagnostics/Registry_diagnostics.xml" in registry
    # Every Fortran unit the library build globs is in the manifest.
    for relative in ("src/framework/mpas_pool_routines.F", "src/core_atmosphere/mpas_atm_dimensions.F",
                     "src/external/SMIOL/smiolf.F90", "src/core_atmosphere/physics/physics_mmm/Makefile.mpas"):
        assert relative in manifest["files"]


def test_the_referee_checkout_has_the_pinned_backend_sources():
    for pin_set, row in nlb.PIN_SETS.items():
        backend = nlb.backend_directory(pin_set)
        assert backend == ROOT / row["checkout"]
        assert all((backend / name).is_file() for name in row["files"])


def _installed(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "native").mkdir(parents=True)
    library = cache / "native" / "lib.so"
    library.write_bytes(b"\x7fELF native")
    reader = cache / "reader"
    reader.write_bytes(b"reader")
    (cache / "tree").mkdir()
    (cache / "thompson").mkdir()
    rb.record_installed(cache, {
        "native-library": {"path": str(library), "sha256": _sha(library)},
        "pool-reader": {"path": str(reader), "sha256": _sha(reader)},
        "mpas-source": {"path": str(cache / "tree"), "sha256": "a" * 64},
        "thompson-tables": {"path": str(cache / "thompson"), "sha256": "b" * 64}})
    return cache


def test_a_key_resolves_to_its_recorded_bytes_and_refuses_changed_ones(tmp_path):
    cache = _installed(tmp_path)
    assert rb.resolve_resource("pool-reader", cache) == cache / "reader"
    assert rb.resolve_resource("mpas-source", cache) == cache / "tree"
    with pytest.raises(rb.PhysicsResourceError, match="physics-resources fetch"):
        rb.resolve_resource("physics-resources", cache)
    with pytest.raises(rb.PhysicsResourceError, match="unknown physics resource key"):
        rb.resolve_resource("libm", cache)
    (cache / "reader").write_bytes(b"other")
    with pytest.raises(rb.PhysicsResourceError, match="changed since installation"):
        rb.resolve_resource("pool-reader", cache)


def test_recording_merges_rows(tmp_path):
    cache = _installed(tmp_path)
    rb.record_installed(cache, {"physics-resources": {"path": str(cache / "reader"), "sha256": _sha(cache / "reader")}})
    rows = rb.read_installed(cache)["resources"]
    assert set(rows) == {"native-library", "pool-reader", "mpas-source", "thompson-tables", "physics-resources"}


def test_first_use_off_names_the_command_and_the_options(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE", "off")
    with pytest.raises(rb.PhysicsResourceError) as error:
        rb.ensure_route_resources(("native-library", "pool-reader"), cache=tmp_path)
    assert "physics-resources build" in str(error.value)
    assert "--native-physics-library" in str(error.value) and "--native-pool-reader" in str(error.value)


def test_the_forecast_door_takes_the_installed_resources(tmp_path, monkeypatch):
    from woof.hex.mpasphys import forecast_v842
    from woof.hex.mpasphys import native_library_build as metadata
    monkeypatch.setattr(metadata, "referee_builder",
                        lambda: pytest.fail("default forecast discovered the native compiler"))
    cache = _installed(tmp_path)
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(rb, "default_cache", lambda: cache)
    for name in ("MPAS_NATIVE_LIBRARY", "MPAS_NATIVE_POOL_READER", "MPAS_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    arguments = argparse.Namespace(native_physics_library=None, native_pool_reader=None, mpas_source=None,
                                   native_physics_tables=None, init=None)
    with pytest.raises(ValueError, match="requires --init"):
        forecast_v842.run_forecast(arguments, None, "")
    # The user route binds only the Rust reader. Registry declarations ship
    # in the wheel, and CUDA resources replace the native table directory.
    assert arguments.native_physics_tables is None


def test_a_run_naming_its_own_tables_keeps_them(tmp_path, monkeypatch):
    from woof.hex.mpasphys import forecast_v842
    cache = _installed(tmp_path)
    monkeypatch.setattr(rb, "default_cache", lambda: cache)
    for name in ("MPAS_NATIVE_LIBRARY", "MPAS_NATIVE_POOL_READER", "MPAS_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    own = [tmp_path / "own-tables"]
    arguments = argparse.Namespace(native_physics_library=None, native_pool_reader=None, mpas_source=None,
                                   native_physics_tables=own, init=None)
    with pytest.raises(ValueError, match="requires --init"):
        forecast_v842.run_forecast(arguments, None, "")
    assert arguments.native_physics_tables == own


def test_empty_cache_uses_the_wheel_reader_before_requiring_init(tmp_path, monkeypatch):
    from woof.hex.mpasphys import forecast_v842
    cache = tmp_path / "empty"
    reader = tmp_path / "wheel-reader"
    reader.write_bytes(b"installed wheel reader")
    monkeypatch.setattr(rb, "default_cache", lambda: cache)
    monkeypatch.setattr(rb, "stage_pool_reader", lambda root, log: {
        "path": str(reader), "sha256": _sha(reader)})
    monkeypatch.setenv("WOOF_HEX_PHYSICS_RESOURCES_FIRST_USE", "off")
    for name in ("MPAS_NATIVE_LIBRARY", "MPAS_NATIVE_POOL_READER", "MPAS_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    arguments = argparse.Namespace(native_physics_library=None, native_pool_reader=None, mpas_source=None,
                                   native_physics_tables=None, init=None)
    with pytest.raises(ValueError, match="requires --init"):
        forecast_v842.run_forecast(arguments, None, "")
    assert set(rb.read_installed(cache)["resources"]) == {"pool-reader"}
    assert rb.resolve_resource("pool-reader", cache) == reader
    assert arguments.native_physics_library is None


def test_a_deck_binds_resource_keys_and_the_receipt_names_them(tmp_path, monkeypatch):
    from woof.hex.verify_mpas import InstalledResources, VerificationRefusal
    cache = _installed(tmp_path)
    resources = InstalledResources(cache)
    argv = [part.format_map({"resource": resources, "out": "/o"})
            for part in ["--native-pool-reader", "{resource:pool-reader}", "--out", "{out}"]]
    assert argv == ["--native-pool-reader", str(cache / "reader"), "--out", "/o"]
    assert resources.used == {"pool-reader": {"path": str(cache / "reader"), "sha256": _sha(cache / "reader")}}
    with pytest.raises(VerificationRefusal, match=r"\{resource:physics-resources\}"):
        "{resource:physics-resources}".format_map({"resource": resources})


def test_stage_library_needs_its_abi_and_receipt(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    library = build / nlb.LIBRARY
    library.write_bytes(b"\x7fELF lib")
    with pytest.raises(nlb.NativeLibraryError, match="abi.json and build-receipt.json"):
        nlb.stage_library(tmp_path / "cache", library)
    (build / "abi.json").write_text("{}")
    (build / "build-receipt.json").write_text(json.dumps({"library_sha256": "0" * 64,
                                                          "abi_sha256": _sha(build / "abi.json")}))
    with pytest.raises(nlb.NativeLibraryError, match="does not match the build receipt"):
        nlb.stage_library(tmp_path / "cache", library)
    (build / "build-receipt.json").write_text(json.dumps({"library_sha256": _sha(library),
                                                          "abi_sha256": _sha(build / "abi.json"),
                                                          "source_pin": rb.MPAS_MODEL_COMMIT}))
    row = nlb.stage_library(tmp_path / "cache", library)
    staged = Path(row["path"])
    assert row["sha256"] == _sha(library) and staged.read_bytes() == library.read_bytes()
    # NativePhysics reads the ABI manifest and receipt beside the library it loads.
    assert (staged.with_name("abi.json")).is_file() and (staged.with_name("build-receipt.json")).is_file()


def test_the_library_build_names_missing_prerequisites(monkeypatch):
    monkeypatch.setattr(nlb.shutil, "which", lambda name: None if name == "gfortran" else "/usr/bin/" + name)
    with pytest.raises(nlb.NativeLibraryError, match="gfortran is not on PATH"):
        nlb.check_prerequisites()


def test_the_reference_stack_is_pinned():
    assert nlb.MPICH["sha256"] == "47d774587a7156a53752218c811c852e70ac44db9c502dc3f399b4cb817e3818"
    assert nlb.PNETCDF["sha256"] == "6f0f7221006c211fce9ddd2c008796b8c69dd717b2ad1be0b4027fc328fd3220"
    script = (ROOT / "tools/native_physics_reference/backend/build_pnetcdf.sh").read_text()
    for flag in ("--disable-fortran", "--disable-cxx", "--disable-shared", "MPICC=mpicc"):
        assert flag in script and flag in nlb.PNETCDF_CONFIGURE


def test_native_compilation_resolves_only_to_the_checkout_referee():
    import inspect
    expected = ROOT / "tools/native_dry_reference/native_library_build.py"
    assert Path(inspect.getsourcefile(nlb.build_native_library)) == expected
    assert nlb.backend_directory() == ROOT / "tools/native_physics_reference/backend"


def test_an_install_refuses_compilation_even_if_a_checkout_tool_was_cached(tmp_path, monkeypatch):
    from woof.hex.mpasphys import native_library_build as metadata
    installed = tmp_path / "site-packages/hexcore/mpasphys/native_library_build.py"
    installed.parent.mkdir(parents=True)
    # Simulate the old wheel's compiler sources. They must not be discovered.
    old_backend = installed.parents[1] / "data/mpasphys-native-backend"
    old_backend.mkdir(parents=True)
    (old_backend / "build.py").write_text("raise AssertionError('packaged compiler ran')\n")
    assert metadata.referee_builder() is nlb
    monkeypatch.setattr(metadata, "__file__", str(installed))
    with pytest.raises(metadata.NativeLibraryError, match="source-checkout referee tool"):
        metadata.build_native_library(tmp_path / "cache")


def _gate_checker():
    spec = importlib.util.spec_from_file_location("check_gate_index", ROOT / "tools" / "check_gate_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_decks_name_route_resources_by_key():
    checker = _gate_checker()
    assert tuple(checker.RESOURCE_KEYS) == rb.RESOURCE_KEYS
    gates = ROOT / "verification" / "gates"
    index = json.loads((gates / "index.json").read_text())
    headline = [entry for entry in index["gates"]["G2"] if "headline" in entry["path"]]
    deck = json.loads((gates / headline[0]["path"]).read_text())
    command = deck["hex_command"]
    for option, key in checker.RESOURCE_OPTIONS.items():
        if option in command:
            assert command[command.index(option) + 1] == "{resource:" + key + "}"
    # Every headline slot is on the GPU, so the deck runs the ported framework: no native library, no MPAS
    # tree (the wheel's Registry), only what `physics-resources fetch` installs.
    assert set(deck["headline_backends"].values()) == {"hex-cuda"}
    assert "--native-physics-library" not in command and "native-library" not in deck["hex_support_inputs"]
    assert set(re.findall(r"\{resource:([^}]*)\}", " ".join(command))) <= {
        "physics-resources", "thompson-tables", "pool-reader"}
    assert checker.check(gates, with_data=False) == []


def test_the_checker_refuses_an_all_gpu_headline_naming_the_native_library(tmp_path):
    checker = _gate_checker()
    deck = {"schema": "mpas-verification-deck-v1", "id": "h", "process_count": 1, "role": "headline",
            "headline_backends": {"pbl": "hex-cuda", "oml": "hex-cuda"},
            "hex_command": ["woof hex", "forecast", "--native-physics-library", "{resource:native-library}"]}
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "h.json").write_text(json.dumps(deck))
    (tmp_path / "index.json").write_text(json.dumps({
        "schema": "mpas-gate-index-v1", "landed_decks": [],
        "gates": {"G2": [{"path": "decks/h.json", "sha256": _sha(tmp_path / "decks" / "h.json")}]}}))
    (tmp_path / "data-manifest.json").write_text(json.dumps({"files": {}}))
    (tmp_path / "ratchet.json").write_text(json.dumps({"schema": "mpas-ratchet-v1", "entries": []}))
    findings = "\n".join(checker.check(tmp_path, with_data=False))
    assert "an all-GPU headline names --native-physics-library" in findings


def test_the_checker_refuses_a_resource_named_by_path(tmp_path):
    checker = _gate_checker()
    deck = {"schema": "mpas-verification-deck-v1", "id": "d", "process_count": 1,
            "hex_command": ["woof hex", "forecast", "--native-physics-library", "{deck_dir}/../data/lib.so",
                            "--mpas-source", "{resource:mpas-tree}"]}
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "d.json").write_text(json.dumps(deck))
    (tmp_path / "index.json").write_text(json.dumps({
        "schema": "mpas-gate-index-v1", "landed_decks": [],
        "gates": {"G2": [{"path": "decks/d.json",
                          "sha256": _sha(tmp_path / "decks" / "d.json")}]}}))
    (tmp_path / "data-manifest.json").write_text(json.dumps({"files": {}}))
    (tmp_path / "ratchet.json").write_text(json.dumps({"schema": "mpas-ratchet-v1", "entries": []}))
    findings = "\n".join(checker.check(tmp_path, with_data=False))
    assert "--native-physics-library names a path, not {resource:native-library}" in findings
    assert "unknown resource key mpas-tree" in findings


def _built_pool_reader() -> Path | None:
    name = rb.POOL_READER + (".exe" if sys.platform == "win32" else "")
    for path in (Path(rb.__file__).resolve().parents[1] / "libexec" / name,
                 ROOT / "rust/physics-pool/target/release" / name):
        if path.is_file():
            return path
    return None


def test_the_pool_transport_takes_the_cpus_it_may_use(monkeypatch, tmp_path):
    # THE BREAKAGE (U-H3, hex 0.4.8): every MPAS-physics member start ran rw_mpas_physics_pool on ONE
    # thread (NLWP 1, 100 % of a core) for ~30 s on the 30 km globe while its card waited.
    from woof.hex.mpasphys import pool_v842 as pool
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(pool, "host_copy_threads", lambda: 24)
    assert pool.pool_transport_threads() == pool.POOL_TRANSPORT_MAX_THREADS == 16
    monkeypatch.setenv("RAYON_NUM_THREADS", "6")       # hexcore's container-quota budget
    assert pool.pool_transport_threads() == 6
    monkeypatch.setattr(pool, "host_copy_threads", lambda: 1)
    assert pool.pool_transport_threads() == 1
    source = (ROOT / "rust/physics-pool/src/bin/rw_mpas_physics_pool.rs").read_text()
    assert f"const MAX_WORKERS: usize = {pool.POOL_TRANSPORT_MAX_THREADS};" in source
    reader = _built_pool_reader()
    if reader is None:
        pytest.skip("no built rw_mpas_physics_pool (the wheel build or cargo makes it)")
    # One worker and sixteen write the same manifest and the same payload bytes.
    fields = [{"pool": "diag", "name": f"f{i}", "kind": "integer" if i % 3 == 0 else "real",
               "dimensions": ["nVertLevels", "nCells"], "shape": [1 + i % 5, 4097 + i],
               "initial": str(i - 20) if i % 3 == 0 else f"{i}.25d0"} for i in range(48)]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema": "mpas-native-pool-request-v1", "sources": [], "fields": fields}))
    outputs = {}
    for workers in (1, 16):
        out = tmp_path / f"w{workers}"
        subprocess.run([str(reader), str(request), str(out)], check=True,
                       env={**os.environ, "RAYON_NUM_THREADS": str(workers)})
        outputs[workers] = out
    manifest = (outputs[1] / "manifest.json").read_bytes()
    assert manifest == (outputs[16] / "manifest.json").read_bytes()
    records = json.loads(manifest)["fields"]
    assert [r["name"] for r in records] == [f["name"] for f in fields]
    for record in records:
        assert (outputs[1] / record["payload"]).read_bytes() == (outputs[16] / record["payload"]).read_bytes()



def test_an_upgraded_wheel_replaces_the_reader_the_cache_recorded(tmp_path, monkeypatch):
    # U-H3: the record kept the reader copied when the cache was filled, so a wheel upgrade left every
    # forecast on the old one-thread transport.  The route takes the reader this wheel carries.
    cache = tmp_path / "cache"
    old = tmp_path / "old-reader"
    old.write_bytes(b"the one-thread reader an earlier wheel staged")
    rb.record_installed(cache, {"pool-reader": {"path": str(old), "sha256": _sha(old)}})
    new = tmp_path / "libexec" / rb.POOL_READER
    new.parent.mkdir()
    new.write_bytes(b"this wheel's reader")
    monkeypatch.setattr(rb, "wheel_pool_reader", lambda: new)
    monkeypatch.setattr(rb, "stage_pool_reader", lambda root, log: {"path": str(new), "sha256": _sha(new)})
    assert rb.ensure_route_resources(("pool-reader",), cache=cache)["pool-reader"] == new
    # A wheel built without a reader (no netCDF-C at build time) keeps the recorded one.
    monkeypatch.setattr(rb, "wheel_pool_reader", lambda: tmp_path / "absent")
    rb.record_installed(cache, {"pool-reader": {"path": str(old), "sha256": _sha(old)}})
    assert rb.ensure_route_resources(("pool-reader",), cache=cache)["pool-reader"] == old


def test_the_rrfs_route_never_fetches_a_registry_file_it_never_reads(tmp_path, monkeypatch):
    # 44c0849's Registry.xml:5142-5144 includes the LGPL-3.0 stochastic_physics/Registry_stoch_physics.xml only
    # under #ifdef STOCHASTIC_PHYSICS. The route's preprocessor never defines it (as the pinned native build),
    # so the first-use source assembly must not download it; every other Registry file is still placed.
    from woof.hex.mpasphys import native_library_build as native
    manifest = native.source_manifest("rrfs-44c0849")
    skipped = [relative for relative in manifest["files"] if relative.startswith(tuple(manifest["run_time_skip"]))]
    assert "src/core_atmosphere/stochastic_physics/Registry_stoch_physics.xml" in skipped
    asked = []

    def fake_fetch(name, origin, path, sha256, target, **_):
        asked.append({"name": name, "origin": origin})
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"")
        return Path(target)

    def fake_fetch_pinned(rows, **_):
        return [fake_fetch(row["name"], row["origin"], row["path"], row["sha256"], row["target"]) for row in rows]

    monkeypatch.setattr(rb, "fetch", fake_fetch)
    if hasattr(rb, "fetch_pinned"):
        monkeypatch.setattr(rb, "fetch_pinned", fake_fetch_pinned)
    rb.assemble_rrfs_source(tmp_path, pin_set="rrfs-44c0849", log=lambda _: None)
    names = {row["name"] for row in asked}
    assert not {row["origin"] for row in asked} & {"stochastic_physics"}
    assert not names & set(skipped)
    assert {"src/core_atmosphere/Registry.xml", "src/core_atmosphere/physics/Registry_tempo.xml",
            "src/core_atmosphere/diagnostics/Registry_diagnostics.xml"} <= names
    assert set(manifest["tables"]) <= names


def test_the_registry_preprocessor_never_reads_an_include_under_an_undefined_macro(tmp_path):
    from woof.hex.mpasphys.pool_v842 import registry_text
    core = tmp_path / "src/core_atmosphere"
    (core / "physics").mkdir(parents=True)
    (core / "physics/Registry_tempo.xml").write_text("<tempo/>\n")
    (core / "Registry.xml").write_text(
        "<registry>\n#ifdef DO_PHYSICS\n#include \"physics/Registry_tempo.xml\"\n"
        "#ifdef STOCHASTIC_PHYSICS\n#include \"stochastic_physics/Registry_stoch_physics.xml\"\n#endif\n"
        "#endif\n</registry>\n")
    assert registry_text(core / "Registry.xml").split() == ["<registry>", "<tempo/>", "</registry>"]
