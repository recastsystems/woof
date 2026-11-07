"""Locating a repository config, wherever this package is running from.

Some case modules are the *module half* of a case whose other half is a
TOML under the repository's top-level ``configs/`` directory.

Through 2.4.1 that directory was in no distribution at all: it is not a
package, and ``[tool.setuptools.packages.find]`` named only ``woof*``,
``tools`` and ``tilestream*``.  A case module that resolves its config as
``Path(__file__).parents[3] / "configs" / NAME`` therefore pointed at
``<site-packages>/configs/NAME`` after ``pip install recast-woof`` -- a path
that had never existed on any machine.  Reading it produced the generic
loader refusal *"... does not exist; pass the experiment .toml that
`woof domain` wrote"*, which is wrong twice over: the wizard does not
emit these configs, and the module that printed it accepted no path to
pass.

Since 2.5.0 ``configs`` ships as a top-level directory of the wheel and
the sdist, so ``<site-packages>/configs/NAME`` is exactly where the file
now is and this module's first-choice path is the one that resolves.  The
resolution ladder below is unchanged and still matters: an environment
that predates 2.5.0, a partial install, or a config a reader keeps outside
the tree all still land in the refusal path, which is why
:func:`shipped_in_wheel` asks the filesystem rather than trusting a
release note.

This module holds the one accurate answer.  :func:`locate` returns the
config when it is really there, and :func:`missing_config_message` builds
the refusal that names *why* it is not there and what the reader can
actually do, given the flag the calling module offers.  Nothing here
knows any case name -- the filename is always an argument.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment override, for a checkout kept somewhere else.  Names a
#: DIRECTORY that holds the config files, standing in for the
#: repository's own ``configs/``.
CONFIG_ROOT_ENV = "WOOF_CONFIGS_ROOT"


def module_name(name: str, spec) -> str:
    """The dotted module name, even under ``python -m``.

    ``__name__`` is ``"__main__"`` in exactly the invocation these case
    modules document, which would make every ``prog=`` and every refusal
    prefix read ``__main__``.  ``__spec__.name`` carries the real one.
    """

    return getattr(spec, "name", None) or name

#: Where the repository's configs live, relative to this package, when
#: this package is being imported out of a source checkout.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]


def config_roots() -> tuple[Path, ...]:
    """Every directory a repository config is looked for, in order."""

    roots: list[Path] = []
    override = os.environ.get(CONFIG_ROOT_ENV)
    if override:
        roots.append(Path(override).expanduser())
    roots.append(_CHECKOUT_ROOT / "configs")
    return tuple(roots)


def locate(name: str) -> Path | None:
    """The readable repository config called ``name``, or ``None``."""

    for root in config_roots():
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def default_path(name: str) -> Path:
    """The path a config WOULD have, for display and for a default.

    Always the checkout-relative location, never the override, so the
    value a module exposes as its default is stable and does not change
    meaning when an environment variable is set.
    """

    return _CHECKOUT_ROOT / "configs" / name


def shipped_in_wheel() -> bool:
    """Whether ``configs/`` is present next to the installed package."""

    return (_CHECKOUT_ROOT / "configs").is_dir()


def missing_config_message(name: str, *, flag: str = "--config") -> str:
    """Why the config is not on this machine, and what to do about it.

    ``flag`` is the option the calling module accepts a path on, so the
    remedy names a door that module really has.
    """

    searched = "\n".join(f"    {root / name}" for root in config_roots())
    lines = [
        f"the repository config {name} is not on this machine.",
        "",
        "Searched:",
        searched,
        "",
    ]
    if not shipped_in_wheel():
        lines += [
            "This case is the module half of a case whose other half is a "
            "TOML under the repository's top-level `configs/` directory.  "
            "gpuwm 2.5.0 and later ship that directory beside the package, "
            "and this install has no copy of it -- so it is either older "
            "than 2.5.0, where `configs/` was in no wheel and no sdist at "
            "all, or it is incomplete.",
            "",
        ]
    lines += [
        "Remedies, either one:",
        f"  * pass the file yourself:  {flag} PATH/TO/{name}",
        f"  * or point {CONFIG_ROOT_ENV} at the directory holding it, and "
        "run without the flag.",
        "",
        "Both want the `configs/` directory of a woof source checkout "
        "(https://github.com/recastsystems/woof); the file is "
        "committed there.  `woof domain` does NOT emit it -- it is a "
        "ratified experiment config, not a wizard product.",
    ]
    return "\n".join(lines)
