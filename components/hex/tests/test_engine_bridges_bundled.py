"""The wheel bundles the three engine bridges no released woof carries.

THE BREAKAGE THIS PREVENTS (release readiness B5, 2026-10-14): the init door
needs rw_mpas_geometry speaking hex-vertical-v1 and the stream I/O needs
rw_mpas_stream.  The frozen 2.9.0 engine bundle carries a geometry bridge
without hex-vertical-v1 and no stream bridge, so a clean install refused the
init door ("lacks hex-vertical-v1") and every native stream read and write
("rw_mpas_stream not found").  The wheel now carries both, pinned by SHA-256
(src/hexcore/data/engine-bridges.json), and the engines ladder reads the
wheel's own woof/hex/libexec before woof's bridge directories.  rw_mpas_static
joined them (2026-10-15): 2.9.0's static builder gives a
generated grid a static declaring binary64 coordinates it does not hold, and
the point door could not build a regional mesh.  These checks need no network,
compiler or device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

import pytest

from woof.hex import engines
from woof.hex.engines import GEOMETRY, EngineRefusal, locate, resolve

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "../../woof/hex/data/engine-bridges.json"


def _renderer_row(directory, binary, **extra):
    """A complete rw_wrfbatch receipt in setup.py's current contract.

    tools/build_s2s_diagnostics.sh is the renderer recipe since the S2S diagnostics
    lane: it builds rw_wrfbatch, its rw_ensbatch sibling and the rw_s2s reducer
    together, and setup.py checks all three by SHA-256 with the reducer's sources.
    """
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sibling = directory / "rw_ensbatch"
    sibling.write_bytes(b"ensemble renderer build")
    reducer = directory / "rw_s2s"
    reducer.write_bytes(b"native reducer build")
    row = {"sha256": sha(binary), "recipe_sha256": sha(ROOT / "tools/build_s2s_diagnostics.sh"),
           "patches": {p.name: sha(p) for p in sorted((ROOT / "docs/s2s/engine-patches").glob("*.patch"))},
           "ensemble_renderer": {"sha256": sha(sibling)},
           "diagnostics": {"rw_s2s_sha256": sha(reducer),
                           "reducer_lock_sha256": sha(ROOT / "tools/rustwx/Cargo.lock"),
                           "reducer_sources": {str(p.relative_to(ROOT)): sha(p) for p in
                                               sorted((ROOT / "tools/rustwx/crates/rw-s2s").rglob("*.rs"))},
                           "renderer_source_sha256": sha(ROOT / "tools/s2s_renderer/s2s_diagnostics.rs")}}
    row.update(extra)
    return row


@pytest.fixture(scope="module")
def hexsetup():
    import setuptools

    original = setuptools.setup
    setuptools.setup = lambda *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("_hex_setup_bridges_under_test", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        setuptools.setup = original
    return module


def test_the_pins_name_both_bridges_their_source_and_where_the_release_fetches_them(hexsetup):
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["schema"] == "gpuwm-hex.engine-bridges/v1"
    assert re.fullmatch(r"[0-9a-f]{40}", pins["gpuwm_source_rev"])
    assert set(pins["bridges"]) == {name.removesuffix(".exe") for name in hexsetup.ENGINE_BRIDGES}
    for name, row in pins["bridges"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), name
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]), name
        # rw_igra2 is crate rw-obs (its bin file is igra2.rs); every other bridge is an rw-mpas bin, a file or
        # (rw_mpas_static since gpuwm 2.9.7) a bin directory whose entry point is main.rs.
        assert row["source"] in (("tools/rustwx/crates/rw-obs/src/bin/igra2.rs",) if name == "rw_igra2"
                                 else (f"tools/rustwx/crates/rw-mpas/src/bin/{name}.rs",
                                       f"tools/rustwx/crates/rw-mpas/src/bin/{name}/main.rs")), name
    assert "hex-vertical-v1" in pins["bridges"]["rw_mpas_geometry"]["protocols"]
    assert "netcdf4" in pins["bridges"]["rw_mpas_stream"]["formats"]
    assert "binary64" in pins["bridges"]["rw_mpas_static"]["holds"]
    # The case-9 port lives in the library; its four sources are pinned beside the bin file.
    assert "case 9" in pins["bridges"]["rw_mpas_lbc"]["holds"]
    assert len(pins["bridges"]["rw_mpas_lbc"]["port_sources"]) == 4
    assert set(pins["release_assets"]) == {"repository", "tag"}
    assert (ROOT / pins["recipe"]).is_file()
    assert (ROOT / "tools/build_netcdf4_static.sh").is_file()
    assert pins["retire_when"]


def _pinned_root(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "tree"
    (root / "src/hexcore/data").mkdir(parents=True)
    pins = {"bridges": {name: {"sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()}}
    (root / "src/hexcore/data/engine-bridges.json").write_text(json.dumps(pins))
    return root


def test_prebuilt_bridges_are_bundled_only_with_their_pinned_bytes(hexsetup, tmp_path, monkeypatch):
    files = {"rw_mpas_geometry": b"\x7fELF geometry", "rw_mpas_stream": b"\x7fELF stream"}
    root = _pinned_root(tmp_path, files)
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    for name, data in files.items():
        (prebuilt / name).write_bytes(data)
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(prebuilt))
    assert sorted(path.name for path in hexsetup.engine_bridges(root)) == sorted(files)
    (prebuilt / "rw_mpas_stream").write_bytes(b"\x7fELF another build")
    with pytest.raises(SystemExit, match="is not the pinned"):
        hexsetup.engine_bridges(root)
    (prebuilt / "rw_mpas_stream").unlink()
    with pytest.raises(SystemExit, match="has no rw_mpas_stream"):
        hexsetup.engine_bridges(root)



@pytest.mark.parametrize("name,marker", [
    ("rw_mpas_stream", "ordered_metadata"),
    ("rw_mpas_static", "--preserve-lake-mask"),
])
def test_hash_matched_old_bridge_cannot_satisfy_required_capability(hexsetup, tmp_path, monkeypatch, name, marker):
    # Old release assets matched their own hashes but stopped forecasts on
    # ordered metadata or stopped TC member preparation on the lake mask.
    data = b"\x7fELF bridge without the required capability"
    root = _pinned_root(tmp_path, {name: data})
    pins_path = root / "src/hexcore/data/engine-bridges.json"
    pins = json.loads(pins_path.read_text())
    required = json.loads(PINS.read_text())["bridges"][name]["required_markers"]
    assert marker in required
    pins["bridges"][name]["required_markers"] = required
    pins_path.write_text(json.dumps(pins))
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    (prebuilt / name).write_bytes(data)
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(prebuilt))
    with pytest.raises(SystemExit, match=f"missing required bridge capability {re.escape(marker)}"):
        hexsetup.engine_bridges(root)


def test_without_a_pinned_directory_or_a_carrying_engine_nothing_is_bundled_and_it_says_so(
        hexsetup, tmp_path, monkeypatch, capsys):
    root = _pinned_root(tmp_path, {"rw_mpas_geometry": b"g", "rw_mpas_stream": b"s", "rw_mpas_static": b"t"})
    monkeypatch.delenv("WOOF_HEX_ENGINE_BRIDGES", raising=False)
    monkeypatch.setattr(hexsetup, "engine_source", lambda: None)
    assert hexsetup.engine_bridges(root) == []
    assert "NOT BUNDLED" in capsys.readouterr().err


def test_source_build_cannot_bundle_a_bridge_from_another_revision(hexsetup, tmp_path, monkeypatch):
    files = {name: ("pinned " + name).encode() for name in hexsetup.ENGINE_BRIDGES}
    root = _pinned_root(tmp_path, files)
    engine = tmp_path / "engine"
    workspace = engine / "tools/rustwx"
    sources = workspace / "crates/rw-mpas"
    (sources / "src/bin").mkdir(parents=True)
    for name in hexsetup.ENGINE_BRIDGES:
        (sources / "src/bin" / (name + ".rs")).write_bytes(b"hex-vertical-v1")
    (sources / "src/static_builder.rs").write_bytes(hexsetup.STATIC_BUILDER_MARK)
    (sources / "src/init").mkdir()
    (sources / "src/init/metfile.rs").write_bytes(hexsetup.LBC_CASE9_MARK)
    (workspace / "crates/rw-obs/src/bin").mkdir(parents=True)
    (workspace / "crates/rw-obs/src/bin/igra2.rs").write_bytes(b"source")
    output = workspace / "target/release"
    output.mkdir(parents=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    monkeypatch.delenv("WOOF_HEX_ENGINE_BRIDGES", raising=False)
    # Cargo emits into the lane scratch override, not workspace/target.
    external = tmp_path / "scratch-target"
    (external / "release").mkdir(parents=True)
    for path in output.iterdir():
        path.rename(external / "release" / path.name)
    output = external / "release"
    monkeypatch.setenv("CARGO_TARGET_DIR", str(external))
    monkeypatch.setattr(hexsetup, "engine_source", lambda: engine)
    monkeypatch.setattr(hexsetup.subprocess, "run", lambda *a, **k: type("Built", (), {"returncode": 0})())
    assert {path.name for path in hexsetup.engine_bridges(root)} == set(files)
    (output / "rw_mpas_stream").write_bytes(b"a successful build from another revision")
    with pytest.raises(SystemExit, match="is not the pinned"):
        hexsetup.engine_bridges(root)


def test_the_wheel_tag_counts_the_bundled_bridges(hexsetup, tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(tmp_path))
    names = {path.name for path in hexsetup.bundled_helpers(ROOT)}
    assert set(hexsetup.ENGINE_BRIDGES) <= names
    assert "rw_wrfbatch" in names


def test_s2s_renderer_prebuilt_rejects_stale_source_or_corrupt_bytes(hexsetup, tmp_path, monkeypatch):
    directory = tmp_path / "bridges"
    directory.mkdir()
    binary = directory / "rw_wrfbatch"
    binary.write_bytes(b"renderer build")
    row = _renderer_row(directory, binary)
    receipt = directory / "provenance.json"
    receipt.write_text(json.dumps({"bridges": {"rw_wrfbatch": row}}))
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(directory))
    assert hexsetup.s2s_renderer_bridge(ROOT) == binary
    binary.write_bytes(b"corrupt build")
    with pytest.raises(SystemExit, match="hash mismatch"):
        hexsetup.s2s_renderer_bridge(ROOT)
    row["patches"] = {}
    receipt.write_text(json.dumps({"bridges": {"rw_wrfbatch": row}}))
    with pytest.raises(SystemExit, match="current S2S fixes"):
        hexsetup.s2s_renderer_bridge(ROOT)


def test_chemistry_converter_prebuilt_matches_source_and_binary(hexsetup, tmp_path, monkeypatch):
    binary = tmp_path / "rw_mpas_convert"
    binary.write_bytes(b"chemistry converter")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = {"sha256": sha(binary),
           "patch_sha256": sha(ROOT / "engine-patches/chemistry-products.patch"),
           "recipe_sha256": sha(ROOT / "tools/build_chemistry_converter.sh")}
    receipt = tmp_path / "provenance.json"
    receipt.write_text(json.dumps({"chemistry_converter": row}))
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(tmp_path))
    assert hexsetup.chemistry_converter_bridge(ROOT) == binary
    binary.write_bytes(b"old converter")
    with pytest.raises(SystemExit, match="current chemistry product provenance"):
        hexsetup.chemistry_converter_bridge(ROOT)
    binary.write_bytes(b"chemistry converter")
    row["patch_sha256"] = "0" * 64
    receipt.write_text(json.dumps({"chemistry_converter": row}))
    with pytest.raises(SystemExit, match="current chemistry product provenance"):
        hexsetup.chemistry_converter_bridge(ROOT)


def test_integrated_renderer_recipe_can_supply_its_actual_provenance(hexsetup, tmp_path, monkeypatch):
    binary = tmp_path / "rw_wrfbatch"
    binary.write_bytes(b"integrated renderer build")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = _renderer_row(tmp_path, binary, patches={},
                        source_copy="unchanged integrated engine source; no patches applied")
    receipt = tmp_path / "provenance.json"

    def write_receipt():
        receipt.write_text(json.dumps({"bridges": {"rw_wrfbatch": row}}))

    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(tmp_path))
    write_receipt()
    assert hexsetup.s2s_renderer_bridge(ROOT) == binary
    row["recipe_sha256"] = "0" * 64
    write_receipt()
    with pytest.raises(SystemExit, match="current S2S fixes"):
        hexsetup.s2s_renderer_bridge(ROOT)
    row["recipe_sha256"] = sha(ROOT / "tools/build_s2s_diagnostics.sh")
    write_receipt()
    binary.write_bytes(b"corrupt integrated renderer")
    with pytest.raises(SystemExit, match="hash mismatch"):
        hexsetup.s2s_renderer_bridge(ROOT)


def test_s2s_renderer_is_reachable_from_the_wheel(tmp_path, monkeypatch):
    from woof.hex.engines import RENDERER
    binary = _executable(tmp_path / "woof/hex/libexec" / RENDERER.name)
    for name in (*RENDERER.env_names, RENDERER.gpuwm_env):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "bundled_candidate", lambda spec, **_: binary)
    assert locate(RENDERER) == (binary, "this distribution's woof/hex/libexec")


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(os.name != "posix", reason="executable bits are a POSIX notion")
def test_the_ladder_reads_the_wheels_libexec_before_gpuwms_bridges(tmp_path, monkeypatch):
    bundled = _executable(tmp_path / "woof/hex/libexec" / GEOMETRY.name)
    staged = _executable(tmp_path / "bridges" / GEOMETRY.name)
    for name in (*GEOMETRY.env_names, GEOMETRY.gpuwm_env):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "bundled_candidate", lambda spec, **_: bundled)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))
    assert resolve(GEOMETRY) == bundled
    assert locate(GEOMETRY) == (bundled, "this distribution's woof/hex/libexec")
    # An explicit setting still wins, and a wheel without the file falls through to woof.
    monkeypatch.setenv(GEOMETRY.env_names[0], str(staged))
    assert resolve(GEOMETRY) == staged
    monkeypatch.delenv(GEOMETRY.env_names[0])
    bundled.unlink()
    assert resolve(GEOMETRY) == staged.resolve()


@pytest.mark.skipif(os.name != "posix", reason="executable bits are a POSIX notion")
def test_a_bundled_bridge_without_its_executable_bit_is_refused_by_name(tmp_path, monkeypatch):
    bundled = _executable(tmp_path / "libexec" / GEOMETRY.name)
    bundled.chmod(0o644)
    for name in (*GEOMETRY.env_names, GEOMETRY.gpuwm_env):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "bundled_candidate", lambda spec, **_: bundled)
    with pytest.raises(EngineRefusal, match="not executable"):
        resolve(GEOMETRY)


def test_a_source_checkout_bundles_nothing_so_its_ladder_is_unchanged():
    # Checkout runs (G0, the gates) resolve exactly as before: the rung finds no file.
    assert not engines.bundled_candidate(GEOMETRY).is_file() or "site-packages" in str(
        engines.bundled_candidate(GEOMETRY))


def test_the_release_build_fetches_checks_and_requires_every_bridge():
    path = ROOT.parent / ".github/workflows/publish.yml"
    if not path.is_file():
        pytest.skip("an sdist carries no .github; the repository checkout does")
    workflow = path.read_text(encoding="utf-8")
    assert "WOOF_HEX_ENGINE_BRIDGES" in workflow and "engine-bridges.json" in workflow
    for name in ("rw_mpas_geometry", "rw_mpas_stream", "rw_mpas_static", "rw_mpas_rrfs_connectivity",
                 "rw_mpas_rrfs_damping", "rw_mpas_rrfs_deformation", "rw_mpas_lbc", "rw_igra2"):
        assert workflow.count(f"--require-libexec {name}") == 2, name
    script = (ROOT / "tools/build_manylinux_artifacts.sh").read_text(encoding="utf-8")
    assert "WOOF_HEX_ENGINE_BRIDGES" in script


def test_source_prebuilts_match_this_tree_so_publish_can_build():
    """publish.yml downloads rw_wrfbatch and rw_sfc_series by pin, with this provenance (source_prebuilts.breakage).

    setup.py refuses a prebuilt whose provenance differs from the tree (s2s_renderer_bridge,
    surface_series_bridge), so a pin that went stale after a recipe, patch or crate edit would stop the
    public build after the tag exists. This test stops it at the commit instead: re-pin from a fresh build.
    """
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    files = pins["source_prebuilts"]["files"]
    # setup.py s2s_renderer_bridge also requires rw_ensbatch and rw_s2s beside rw_wrfbatch.
    assert set(files) == {"rw_wrfbatch", "rw_ensbatch", "rw_s2s", "rw_sfc_series", "rw_mpas_convert"}
    # setup.py chemistry_converter_bridge refuses a converter whose patch or recipe differs from the tree;
    # publish.yml cannot build it from the sparse engine checkout, so it rides the bridge release too.
    converter = files["rw_mpas_convert"]["provenance"]
    assert converter["patch_sha256"] == hashlib.sha256((ROOT / "engine-patches/chemistry-products.patch").read_bytes()).hexdigest()
    assert converter["recipe_sha256"] == hashlib.sha256((ROOT / "tools/build_chemistry_converter.sh").read_bytes()).hexdigest()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, row in files.items():
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and row["provenance"]["sha256"] == row["sha256"]
        assert isinstance(row["bytes"], int) and row["bytes"] > 0
    renderer = files["rw_wrfbatch"]["provenance"]
    # The recipe setup.py checks and build_engine_bridges.sh runs.
    assert renderer["recipe_sha256"] == sha(ROOT / "tools/build_s2s_diagnostics.sh")
    assert renderer["patches"] == {p.name: sha(p) for p in sorted((ROOT / "docs/s2s/engine-patches").glob("*.patch"))}
    assert renderer["ensemble_renderer"]["sha256"] == files["rw_ensbatch"]["sha256"]
    diagnostics = renderer["diagnostics"]
    assert diagnostics["rw_s2s_sha256"] == files["rw_s2s"]["sha256"]
    assert diagnostics["reducer_lock_sha256"] == sha(ROOT / "tools/rustwx/Cargo.lock")
    assert diagnostics["reducer_sources"] == {
        str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "tools/rustwx/crates/rw-s2s").rglob("*.rs"))}
    assert diagnostics["renderer_source_sha256"] == sha(ROOT / "tools/s2s_renderer/s2s_diagnostics.rs")
    crate = ROOT / "tools/rustwx/crates/rw-sfc-series"
    assert files["rw_sfc_series"]["provenance"]["sources"] == {
        str(p.relative_to(crate)).replace(os.sep, "/"): sha(p) for p in sorted(crate.rglob("*")) if p.is_file()}
