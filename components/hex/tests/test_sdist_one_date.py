"""The publish build writes the source archive again with one date, so two builds of one commit are one sdist.

THE BREAKAGE (dist47 of hex 0.6.0, 2026-11-15): two release builds of one commit gave two source archives with the
same 4,340 files byte for byte and two sha256s; 13 members carried the build's clock.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "sdist_one_date.py"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "publish.yml"
EPOCH = 1767225600


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("_sdist_one_date_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _archive(path: Path, when: int, order: list[str]) -> None:
    files = {"pkg-1.0/PKG-INFO": b"Name: pkg\n", "pkg-1.0/src/a.py": b"print(1)\n", "pkg-1.0/setup.cfg": b"[x]\n"}
    with tarfile.open(path, "w:gz") as tf:
        d = tarfile.TarInfo("pkg-1.0")
        d.type, d.mode, d.mtime = tarfile.DIRTYPE, 0o755, when
        tf.addfile(d)
        for name in order:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode, ti.mtime, ti.uid, ti.uname = len(files[name]), 0o644, when + len(name), 1000, "builder"
            tf.addfile(ti, io.BytesIO(files[name]))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_two_builds_with_other_clocks_become_one_file(tool, tmp_path):
    a, b = tmp_path / "a" / "pkg-1.0.tar.gz", tmp_path / "b" / "pkg-1.0.tar.gz"
    a.parent.mkdir(), b.parent.mkdir()
    _archive(a, 1_800_000_000, ["pkg-1.0/PKG-INFO", "pkg-1.0/src/a.py", "pkg-1.0/setup.cfg"])
    _archive(b, 1_800_000_777, ["pkg-1.0/setup.cfg", "pkg-1.0/PKG-INFO", "pkg-1.0/src/a.py"])
    assert _sha(a) != _sha(b)
    assert tool.one_date(a, EPOCH) == tool.one_date(b, EPOCH) == _sha(a) == _sha(b)
    with tarfile.open(a) as tf:
        got = {m.name: (m.mtime, m.uid, m.uname, tf.extractfile(m).read() if m.isfile() else None)
               for m in tf.getmembers()}
    assert got["pkg-1.0/src/a.py"] == (EPOCH, 0, "", b"print(1)\n")
    assert {v[0] for v in got.values()} == {EPOCH}


def test_the_publish_build_runs_it_on_the_sdist_with_the_wheel_date():
    if not WORKFLOW.is_file():
        pytest.skip("publish.yml is outside this tree (sdist install)")
    lines = [line.strip() for line in WORKFLOW.read_text(encoding="utf-8").splitlines()]
    assert '"$PY" hex/tree/tools/sdist_one_date.py --epoch "${WHEEL_DATE_EPOCH}" raw/*.tar.gz' in lines
    at = lines.index('"$PY" hex/tree/tools/sdist_one_date.py --epoch "${WHEEL_DATE_EPOCH}" raw/*.tar.gz')
    assert at < lines.index("cp raw/*.tar.gz dist/"), "the archive is rewritten before it is copied to dist"
