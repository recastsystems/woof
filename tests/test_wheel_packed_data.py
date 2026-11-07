"""The wheels carry four large data members xz-compressed (woof.packed_data).

Breakage this prevents (WOOF 1.1.6): the platform wheel built at 105,480,112
bytes against PyPI's 100,000,000-byte per-file limit.  setup.py re-encodes the
members woof.packed_data.WHEEL_XZ_DATA names as ``<name>.xz`` in every wheel,
and their readers take either form.  These tests hold the three halves
together: the list names real tree files, the build step writes exactly
those as xz and nothing else, and a reader given only the ``.xz`` member
returns the same bytes.
"""

from __future__ import annotations

import importlib.util
import json
import lzma
from pathlib import Path
import shutil

import pytest

from woof import packed_data

REPO = Path(__file__).resolve().parents[1]


def _setup_module(monkeypatch):
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    spec = importlib.util.spec_from_file_location("_woof_setup_under_test", REPO / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_listed_member_is_a_tree_file():
    for relative in packed_data.WHEEL_XZ_DATA:
        assert (REPO / relative).is_file(), relative
        assert not relative.endswith(packed_data.SUFFIX), relative


def test_setup_reads_the_same_list(monkeypatch):
    setup = _setup_module(monkeypatch)
    assert setup.wheel_xz_data() == packed_data.WHEEL_XZ_DATA


def test_the_build_step_packs_exactly_the_listed_members(monkeypatch, tmp_path):
    setup = _setup_module(monkeypatch)
    payload = {}
    for index, relative in enumerate(packed_data.WHEEL_XZ_DATA):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload[relative] = (f"member {index} " * 500).encode()
        path.write_bytes(payload[relative])
    bystander = tmp_path / "woof" / "verify" / "other.json"
    bystander.parent.mkdir(parents=True, exist_ok=True)
    bystander.write_text("{}", encoding="utf-8")
    rows = setup.pack_wheel_xz_data(tmp_path)
    assert {row[0] for row in rows} == set(packed_data.WHEEL_XZ_DATA)
    for relative, raw in payload.items():
        plain = tmp_path / relative
        assert not plain.exists(), relative
        assert lzma.decompress(packed_data.packed(plain).read_bytes()) == raw
        assert packed_data.read_bytes(plain) == raw
        assert packed_data.size(plain) == len(raw)
    assert bystander.is_file() and not packed_data.packed(bystander).exists()


def test_a_reader_prefers_the_plain_file_and_falls_back_to_xz(tmp_path):
    plain = tmp_path / "compiler-recipes.json"
    with pytest.raises(FileNotFoundError):
        packed_data.read_text(plain)
    packed_data.packed(plain).write_bytes(lzma.compress(b"[1]"))
    (tmp_path / "compiler-recipes-cuda12.json").write_text("[2]", encoding="utf-8")
    assert packed_data.read_text(plain) == "[1]"
    assert packed_data.glob(tmp_path, "compiler-recipes*.json") == [
        tmp_path / "compiler-recipes-cuda12.json", plain]
    plain.write_text("[3]", encoding="utf-8")
    assert packed_data.read_text(plain) == "[3]"
    assert packed_data.glob(tmp_path, "compiler-recipes*.json") == [
        tmp_path / "compiler-recipes-cuda12.json", plain]


def test_the_registry_history_reads_from_its_xz_member(monkeypatch, tmp_path):
    from woof import physics_registry

    source = REPO / "woof" / "physics_registry_history.json"
    expected = json.loads(source.read_text(encoding="utf-8"))
    target = tmp_path / source.name
    packed_data.packed(target).write_bytes(lzma.compress(source.read_bytes()))
    monkeypatch.setattr(physics_registry, "REGISTRY_PHYSICS_HISTORY_PATH", target)
    monkeypatch.setattr(physics_registry, "_registry_physics_history", None)
    assert physics_registry.registry_physics_history() == expected


def test_the_udm_allocation_is_sized_without_the_table_file(monkeypatch, tmp_path):
    from woof.core import udm
    from woof.core.preflight import thompson_generation_coefficient_bytes

    shutil.copy(udm.DATA / "identity.json", tmp_path / "identity.json")
    monkeypatch.setattr(udm, "DATA", tmp_path)

    class _Config:
        mp_physics = 27

    expected = json.loads((tmp_path / "identity.json").read_text())["bytes"] + 32768
    assert thompson_generation_coefficient_bytes(_Config()) == expected
    assert expected == (REPO / "woof/data/udm/udm-tables.bin").stat().st_size + 32768
