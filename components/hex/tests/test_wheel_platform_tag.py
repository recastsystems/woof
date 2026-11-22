"""The wheel's platform tag is measured from the helper it carries.

THE BREAKAGE: 0.3.4's wheel was tagged cp3X-cp3X-linux_x86_64.  PyPI rejects
a bare linux_x86_64 tag, so the upload would have failed, and the cp3X tag
tied the wheel to the build's Python minor although the bundled
hex_contract_hostprep is a subprocess executable linking no libpython.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

import site_support

SETUP = Path(__file__).resolve().parents[1] / "setup.py"


@pytest.fixture(scope="module")
def hexsetup():
    setuptools = site_support.setuptools_or_skip("loading setup.py's wheel tag")

    original = setuptools.setup
    setuptools.setup = lambda *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("_hex_setup_under_test", SETUP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        setuptools.setup = original
    return module


def test_the_glibc_floor_is_the_newest_version_the_binary_names(hexsetup, tmp_path):
    binary = tmp_path / "helper"
    binary.write_bytes(b"\x7fELF..GLIBC_2.2.5\0GLIBC_2.34\0GLIBC_2.17\0..")
    assert hexsetup.glibc_floor(binary) == (2, 34)
    binary.write_bytes(b"\x7fELF static, no versioned glibc symbols")
    assert hexsetup.glibc_floor(binary) is None


def test_the_tag_never_claims_a_bare_linux_platform(hexsetup, tmp_path, monkeypatch):
    binary = tmp_path / "helper"
    binary.write_bytes(b"GLIBC_2.28\0GLIBC_2.34\0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hexsetup.platform, "machine", lambda: "x86_64")
    assert hexsetup.wheel_platform_tag(binary) == "manylinux_2_34_x86_64"
    binary.write_bytes(b"no glibc at all")
    assert hexsetup.wheel_platform_tag(binary) == "manylinux_2_17_x86_64"


def test_the_floor_is_the_newest_over_every_bundled_elf(hexsetup, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hexsetup.platform, "machine", lambda: "x86_64")
    helper, decoder, absent = tmp_path / "helper", tmp_path / "decoder", tmp_path / "absent"
    helper.write_bytes(b"GLIBC_2.17\0")
    decoder.write_bytes(b"GLIBC_2.2.5\0GLIBC_2.28\0")
    assert hexsetup.wheel_platform_tag(helper, decoder, absent) == "manylinux_2_28_x86_64"


def test_the_wheel_is_py3_none(hexsetup, monkeypatch):
    # Measured by calling get_tag, not by reading setup.py's text: the 0.4.0
    # version matched one source line, so adding the physics-table decoders
    # to the floor (H2) failed it while the tag stayed py3-none-manylinux.
    assert hexsetup.HexWheel.get_tag.__qualname__ == "HexWheel.get_tag"
    measured = []

    def tag(binary, *others):
        measured.append((binary, *others))
        return "manylinux_2_28_x86_64"

    monkeypatch.setattr(hexsetup, "wheel_platform_tag", tag)
    monkeypatch.setattr(hexsetup.Path, "is_file", lambda self: True)
    wheel = object.__new__(hexsetup.HexWheel)
    assert hexsetup.HexWheel.get_tag(wheel) == ("py3", "none", "manylinux_2_28_x86_64")
    names = [path.name for path in measured[0]]
    # The floor is measured over the helper and every ELF bundled beside it.
    assert names[0] == hexsetup.HOSTPREP
    assert hexsetup.OBS_SCORER in names
    assert hexsetup.NETCDF_COPY in names
    assert set(hexsetup.PHYSICS_TABLES) <= set(names)
    assert hexsetup.POOL_READER in names
    assert hexsetup.SURFACE_READER in names


@pytest.mark.parametrize("corrupt", [None, "library", "source"])
def test_prebuilt_surface_reader_requires_matching_library_and_source(
    hexsetup, tmp_path, monkeypatch, corrupt
):
    import hashlib
    import json

    root = tmp_path / "source"
    source = root / "tools/rustwx/crates/rw-surface/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reader source")
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    library = prebuilt / hexsetup.SURFACE_READER
    library.write_bytes(b"reader library")
    record = {"sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (prebuilt / "provenance.json").write_text(json.dumps({"surface_reader": record}))
    monkeypatch.setenv("WOOF_HEX_ENGINE_BRIDGES", str(prebuilt))
    if corrupt:
        (library if corrupt == "library" else source).write_bytes(b"changed")
        with pytest.raises(SystemExit, match="surface reader build provenance differs"):
            hexsetup.BuildPy.build_surface_reader(root)
    else:
        assert hexsetup.BuildPy.build_surface_reader(root) == library
