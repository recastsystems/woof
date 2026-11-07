"""Build hook: carry woof's verification oracles beside the kernel packs.

Since 2.9.12 the native-WRF captures under ``woof/data/<scheme>/oracle``
(and their siblings, listed in ``woof.data_assets.VERIFY_ORACLE_TREES``)
ship in this distribution instead of the ``woof`` wheel.  The reason is
the 100,000,000-byte per-file limit: the 2.9.12 ``woof`` manylinux wheel
measured 109,246,370 bytes with them and ``recast-woof-data`` had 1.37 MB of room,
so they moved to the one required companion with space for them.
``woof.data_assets.verify_oracle_path`` reads them from here on an install.

The repository keeps ONE copy, in ``woof/data``, where the oracle builders
write and every test reads.  This hook copies those trees into the build:

* ``build_py`` (a wheel built straight from the checkout) copies them into
  ``build/lib/gpuwm_kernels/verify_oracles/``;
* ``sdist`` copies them into the source archive at the same path, so a wheel
  built from the archive (``python -m build``'s default route) finds them
  through the ``verify_oracles/**/*`` package-data glob.

Either way ``verify_oracles/MANIFEST.json`` records every file's size and
SHA-256, and a build that can find the trees neither beside the checkout
nor inside the archive is refused: a companion without them would turn
``woof verify`` into a missing-file error on every install.
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _BuildPy
from setuptools.command.sdist import sdist as _Sdist

HERE = Path(__file__).resolve().parent
PACKAGE = "gpuwm_kernels"
SUBDIR = "verify_oracles"
MANIFEST = "MANIFEST.json"
SCHEMA = "gpuwm-kernels.verify-oracles.v1"
_SKIP = {"__pycache__"}


def engine_trees() -> tuple[Path, tuple[str, ...]] | None:
    """The sibling engine checkout's data root and oracle trees, or None."""

    declaration = HERE.parent / "woof" / "data_assets.py"
    if not declaration.is_file():
        return None
    for node in ast.parse(declaration.read_text(encoding="utf-8")).body:
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else node.targets[0] if isinstance(node, ast.Assign) else None)
        if getattr(target, "id", None) == "VERIFY_ORACLE_TREES":
            return HERE.parent / "woof" / "data", tuple(ast.literal_eval(node.value))
    raise SystemExit("woof/data_assets.py declares no VERIFY_ORACLE_TREES; "
                     "cannot tell which verification oracles this companion carries")


def stage(destination: Path) -> dict:
    """Copy every oracle tree into ``destination`` and write its manifest."""

    found = engine_trees()
    assert found is not None
    root, trees = found
    if destination.exists():
        shutil.rmtree(destination)
    files = {}
    for tree in trees:
        source = root / tree
        if not source.is_dir():
            raise SystemExit(f"verification oracle tree woof/data/{tree} is missing "
                             "from the checkout; the companion cannot be built without it")
        for path in sorted(source.rglob("*")):
            if path.is_dir() or _SKIP.intersection(path.relative_to(root).parts) or path.suffix == ".pyc":
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"schema": SCHEMA, "trees": list(trees), "files": files}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return manifest


def check(directory: Path) -> None:
    """Refuse a staged copy that is not exactly what its manifest records."""

    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(
            "gpuwm-kernels: no verification oracles to package -- neither a sibling woof "
            "checkout (woof/data_assets.py) nor gpuwm_kernels/verify_oracles/MANIFEST.json "
            "from a source archive.  Build from the repository or from its sdist.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, row in manifest["files"].items():
        data = (directory / relative).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"gpuwm-kernels: verify_oracles/{relative} does not match its manifest")


class BuildPy(_BuildPy):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / PACKAGE / SUBDIR
        if engine_trees() is not None:
            stage(destination)
        check(destination)


class Sdist(_Sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / PACKAGE / SUBDIR
        if engine_trees() is not None:
            stage(destination)
        check(destination)


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
