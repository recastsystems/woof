"""Package data the wheels carry xz-compressed, and the one reader for it.

Why: PyPI refuses any file of 100,000,000 bytes or more.  The WOOF 1.1.6
platform wheel built at 105,480,112 bytes (engine 2.9.12 grew the package
data by about 8 MB over 1.1.5).  Four members of it are large and highly
redundant -- two hex compiler-recipe catalogs, the legacy registry-physics
history and the UDM (mp=27) initialized tables -- and xz keeps 10.3 MB less
of them than a wheel's deflate does (measured on the 1.1.6 pure wheel: 15.5 MB
of history JSON is 3.02 MB deflated, 0.11 MB xz).

So :data:`WHEEL_XZ_DATA` names them, ``setup.py`` (``build_py``) writes
each one into every wheel as ``<name>.xz`` in place of ``<name>``, and the
code that reads them reads through :func:`read_bytes` / :func:`read_text`,
which take the plain file when it is there (a checkout, an editable
install, an sdist tree) and the ``.xz`` member otherwise (a wheel install).
The bytes a reader gets are identical either way; every digest and size
check stays on those bytes.  The source tree, the sdist and every test
that reads the tree keep the plain files.
"""
from __future__ import annotations

import lzma
from pathlib import Path

#: Project-relative paths the wheels store as ``<path>.xz``.  ``setup.py``
#: reads this tuple with ``ast`` (it cannot import woof), so it stays a literal.
WHEEL_XZ_DATA: tuple[str, ...] = (
    "woof/physics_registry_history.json",
    "woof/hex/data/compiler-recipes.json",
    "woof/hex/data/compiler-recipes-cuda12.json",
    "woof/data/udm/udm-tables.bin",
)

SUFFIX = ".xz"


def packed(path: Path | str) -> Path:
    """The ``.xz`` member a wheel install carries for ``path``."""

    path = Path(path)
    return path.with_name(path.name + SUFFIX)


def located(path: Path | str) -> Path | None:
    """``path`` when it exists, else its ``.xz`` member when that exists, else None."""

    path = Path(path)
    if path.is_file():
        return path
    member = packed(path)
    return member if member.is_file() else None


def exists(path: Path | str) -> bool:
    return located(path) is not None


def read_bytes(path: Path | str) -> bytes:
    """The bytes of ``path``, from the plain file or from its ``.xz`` member.

    A missing file raises :class:`FileNotFoundError` naming the plain path.
    """

    path = Path(path)
    found = located(path)
    if found is None:
        raise FileNotFoundError(f"{path} (nor {packed(path).name}) is not installed")
    if found == path:
        return path.read_bytes()
    return lzma.decompress(found.read_bytes())


def read_text(path: Path | str, encoding: str = "utf-8") -> str:
    return read_bytes(path).decode(encoding)


def size(path: Path | str) -> int:
    """The byte size of the data ``path`` holds (decompressed for an ``.xz`` member)."""

    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return len(read_bytes(path))


def glob(directory: Path | str, pattern: str) -> list[Path]:
    """``directory.glob(pattern)``, each ``.xz`` member reported under its plain name.

    A wheel install holds ``compiler-recipes.json.xz`` where a checkout holds
    ``compiler-recipes.json``; both answer ``compiler-recipes*.json`` with the
    plain path, which :func:`read_text` then reads.
    """

    directory = Path(directory)
    found = {path for path in directory.glob(pattern) if path.is_file()}
    for member in directory.glob(pattern + SUFFIX):
        if member.is_file():
            found.add(member.with_name(member.name[:-len(SUFFIX)]))
    return sorted(found)
