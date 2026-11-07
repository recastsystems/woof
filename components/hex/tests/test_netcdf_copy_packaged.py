"""The vertical door's CDF-5 copier ships in the wheel and is found there first.

THE BREAKAGE (verify-final-hex item 4, 2026-10-12): ``gpuwm-hex-netcdf-copy``
was looked up only through an environment variable, PATH and a source tree's
``rust/netcdf-copy``.  The wheel did not carry it and the sdist did not carry
the crate, so every pip install copied vertical templates the slow way (about
an hour on a 1.5 M-cell mesh) and refused meshes past the CDF-2 variable limit
with "Build rust/netcdf-copy", which no pip user can do.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[1]


def test_the_sdist_carries_the_copier_crate_and_not_its_build_output(monkeypatch):
    filelist = pytest.importorskip("setuptools._distutils.filelist")
    files = filelist.FileList()
    monkeypatch.chdir(TREE)  # the template engine reads paths relative to the sdist root
    files.findall(".")
    for line in (TREE / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            files.process_template_line(line)
    shipped = {name.replace("\\", "/").lstrip("./") for name in files.files}
    assert "rust/netcdf-copy/Cargo.toml" in shipped
    assert "rust/netcdf-copy/Cargo.lock" in shipped
    assert "rust/netcdf-copy/src/main.rs" in shipped
    assert "rust/netcdf-copy/build.rs" in shipped
    assert not any(name.startswith("rust/netcdf-copy/target/") for name in shipped)


def test_the_wheel_build_installs_the_copier_into_libexec():
    source = (TREE / "setup.py").read_text(encoding="utf-8")
    assert 'NETCDF_COPY = "gpuwm-hex-netcdf-copy"' in source
    # Every helper now uses the same-file-safe copier when build output is reused.
    assert "copy_helper(copier, destination.parent / NETCDF_COPY)" in source
    # the wheel's tag accounts for the copier's glibc floor too
    assert '"rust/netcdf-copy/target/release" / NETCDF_COPY' in source


def test_the_door_finds_the_packaged_copier_before_path_and_cargo(tmp_path, monkeypatch):
    from woof.hex import vertical_spec

    package = tmp_path / "hexcore"
    (package / "libexec").mkdir(parents=True)
    copier = package / "libexec" / ("gpuwm-hex-netcdf-copy" + (".exe" if os.name == "nt" else ""))
    copier.write_bytes(b"")
    copier.chmod(0o755)
    monkeypatch.setattr(vertical_spec, "__file__", str(package / "vertical_spec.py"))
    from woof.hex import helpers  # (WOOF) the wheel's libexec is read through woof.hex.helpers
    monkeypatch.setattr(helpers, "PACKAGED_DIR", package / "libexec")
    monkeypatch.delenv("WOOF_HEX_NETCDF_COPY", raising=False)
    monkeypatch.setattr(vertical_spec.shutil, "which", lambda name: pytest.fail(f"looked on PATH for {name}"))
    assert vertical_spec._netcdf_copy_executable() == str(copier)


def test_an_explicit_copier_still_wins(tmp_path, monkeypatch):
    from woof.hex import vertical_spec

    monkeypatch.setenv("WOOF_HEX_NETCDF_COPY", str(tmp_path / "mine"))
    assert vertical_spec._netcdf_copy_executable() == str(tmp_path / "mine")


def test_a_netcdf4_template_is_copied_as_it_is(tmp_path, monkeypatch):
    from woof.hex import vertical_spec

    source = tmp_path / "static.nc"
    source.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    monkeypatch.setattr(vertical_spec, "_netcdf_copy_executable",
                        lambda: pytest.fail("a netCDF-4 template went to the classic-only copier"))
    out = tmp_path / "copy.nc"
    assert vertical_spec._copy_vertical_template(source, out, n_edges=10, n_levels=5) == "source-format-copy"
    assert out.read_bytes() == source.read_bytes()


def test_the_fallback_message_no_longer_blames_a_missing_cargo():
    text = (TREE / "../../woof/hex/vertical_spec.py").read_text(encoding="utf-8")
    assert "there is no cargo to build" not in text
    assert "woof/hex/libexec, not on PATH" in text


def test_the_wheel_carries_the_static_netcdf_notice():
    notice = TREE / "../../woof/hex/data/licenses/netcdf-c-COPYRIGHT"
    assert "Copyright 2018 Unidata" in notice.read_text(encoding="utf-8")
    script = (TREE / "tools/build_netcdf_classic_static.sh").read_text(encoding="utf-8")
    assert "a474149844e6144566673facf097fea253dc843c37bc0a7d3de047dc8adda5dd" in script
    assert "--disable-shared" in script and "--disable-netcdf-4" in script
