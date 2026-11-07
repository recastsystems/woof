"""A193: the literal-division gate reads sources only from the tree it scans.

The breakage this prevents: on the 2.8.1 B200 bench woof was installed
from a wheel beside a source tests tree, and the compiler census
(tools/literal_division_census.py) took p3_device's sources from the
imported package, outside the tree it scanned, and stopped on a ValueError
before compiling a unit: three test_literal_division_gate failures that
were no division defect.  A census or scan reading the imported package
instead of the scanned tree gates sources other than the ones under test.

Here the scanned tree is a copy of this tree's kernels, outside the
imported woof, with a marker appended to every kernel source.  Every unit
the census builds and every offender the scans report must come from that
copy, and no file outside it may be opened.  CPU only; nothing compiles.
"""
from __future__ import annotations

import builtins
import os
import pathlib
import shutil
from pathlib import Path

import pytest

from tools import literal_division_census as census
from tools import literal_division_scan as scan

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "// A193 scanned-tree marker"


@pytest.fixture
def scanned_tree(tmp_path):
    """A kernel tree outside the imported package, every source marked."""
    root = tmp_path / "scanned"
    kernels = scan.kernel_dir(root)
    shutil.copytree(scan.kernel_dir(REPO_ROOT), kernels,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for path in list(kernels.glob("*.cu")) + list(kernels.glob("*.cuh")):
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "\n" + MARKER + "\n", encoding="utf-8")
    # Copy the exact declared coefficient headers and their authority.
    from woof.core.kernels import EXTRA_HEADERS, _TABLE_HEADERS
    for relative, _size, _digest in _TABLE_HEADERS.values():
        target = root / "woof" / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / "woof" / "data" / relative, target)
    for headers in EXTRA_HEADERS.values():
        for header in headers:
            if header.startswith("../../data/"):
                target = (kernels / header).resolve()
                relative = target.relative_to(root.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO_ROOT / relative, target)
                shutil.copyfile((REPO_ROOT / relative).parent / "MANIFEST.json",
                                target.parent / "MANIFEST.json")
    # Parameter headers are source inputs too; copy the registered assets.
    from woof.table_assets import kernel_parameter_source_pieces
    for unit in kernels.glob("*.cu"):
        for label, _ in kernel_parameter_source_pieces(unit.stem):
            target = root / label
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / label, target)
    # pxsfclay's constant prefix is part of its unit (kernels.module_source).
    target = root / "woof" / "data" / "pxsfclay" / "constants.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "woof" / "data" / "pxsfclay" / "constants.json", target)
    return root.resolve()


def _record_reads(monkeypatch) -> list[Path]:
    """Every file opened for reading from here on, by Path or open()."""
    opened: list[Path] = []
    path_open, plain_open = pathlib.Path.open, builtins.open

    def recorded_path_open(self, mode="r", *args, **kwargs):
        if "r" in mode or "+" in mode:
            opened.append(Path(os.path.realpath(self)))
        return path_open(self, mode, *args, **kwargs)

    def recorded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and (
                "r" in mode or "+" in mode):
            opened.append(Path(os.path.realpath(file)))
        return plain_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", recorded_path_open)
    monkeypatch.setattr(builtins, "open", recorded_open)
    return opened


def _outside(opened, root) -> list[str]:
    return sorted({str(path) for path in opened
                   if not path.is_relative_to(root)})


def test_the_census_builds_every_unit_from_the_scanned_tree(
        monkeypatch, scanned_tree):
    reference = {unit.key: unit for unit in census.production_units(REPO_ROOT)}
    assert not any(MARKER in unit.source for unit in reference.values())
    opened = _record_reads(monkeypatch)
    units = census.production_units(scanned_tree)
    assert opened, "the census opened no source"
    assert not _outside(opened, scanned_tree), _outside(opened, scanned_tree)
    assert [unit.key for unit in units] == list(reference)
    for unit in units:
        assert MARKER in unit.source, unit.key
        assert unit.options == reference[unit.key].options, unit.key
    # p3_device is labelled by this tree's paths, the label the B200 bench
    # could not form from an installed package.
    (p3,) = [unit for unit in units if unit.key == "p3_device"]
    labels = {label for _start, label, _first in p3.segments}
    assert {"woof/core/kernels/noahmp_leaves.cu",
            "woof/core/kernels/p3.cu"} <= labels
    # The dedicated urban loader needs its glibc/trig/layout headers. A bare
    # kernel failed compilation and silently omitted this production route.
    (urban,) = [unit for unit in units if unit.key == "kernels:urban_bep_bem"]
    assert "struct ubm_Context" in urban.source and "sincosf" in urban.source


def test_the_scans_report_and_read_only_the_scanned_tree(
        monkeypatch, scanned_tree):
    # This gate measures scan-root isolation. The separate source gate
    # judges each production finding. Preserve them here while requiring
    # the planted defect to appear from this tree as well.
    reference_kernel = scan.kernel_file_offenders(REPO_ROOT)
    reference_inline = scan.inline_kernel_offenders(REPO_ROOT)
    kernels = scan.kernel_dir(scanned_tree)
    (kernels / "planted_a193.cu").write_text(
        'extern "C" __global__ void k(float* y)\n'
        "{ y[0] = y[1] / 3.0f; }\n", encoding="utf-8")
    (scanned_tree / "woof" / "planted_a193.py").write_text(
        'BODY = r"""\nextern "C" __global__ void k(float* y)\n'
        '{ y[0] = y[1] / 7.0f; }\n"""\n', encoding="utf-8")
    opened = _record_reads(monkeypatch)
    assert sorted(scan.kernel_file_offenders(scanned_tree)) == sorted([
        *reference_kernel,
        "woof/core/kernels/planted_a193.cu:2 divides by 3.0f"])
    assert sorted(scan.inline_kernel_offenders(scanned_tree)) == sorted([
        *reference_inline,
        "woof/planted_a193.py:1 (string line 3) divides by 7.0f"])
    assert not _outside(opened, scanned_tree), _outside(opened, scanned_tree)
    assert scan.kernel_dir(scanned_tree) / "common.cuh" in opened


def test_a_root_without_kernel_sources_is_refused_not_passed(tmp_path):
    """An empty scan would pass the gate on no sources at all."""
    with pytest.raises(FileNotFoundError):
        scan.kernel_file_offenders(tmp_path)
    with pytest.raises(FileNotFoundError):
        scan.inline_kernel_offenders(tmp_path)
    with pytest.raises(FileNotFoundError):
        census.production_units(tmp_path)
