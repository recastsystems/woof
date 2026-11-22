"""A runtime interlock that makes the import constraint structural.

The forecast worker must import ``woof`` only from the Arwen checkout
the port pins.  Relying on nobody ever typing ``from woof.cycle.anchor
import read_anchor`` inside the worker is not a constraint, it is a
hope; the first contributor who needs an anchor helper will type it, the
port will refuse eight frames deep inside ``_construct_device_stack``
with a message about frozen Arwen, and whoever debugs that will lose an
evening.

So the worker installs this guard on ``sys.meta_path`` before it touches
anything, and the guard refuses the import at the point of the import,
naming the bridge and the reason.  Two rules:

``woof.cycle`` and everything under it is refused ALWAYS.
    The spine namespace can never legitimately appear in a forecast
    process.  The pinned Arwen checkout is a frozen v2 tree whose
    manifest is entirely ``woof/core/...``; it has no cycling spine.  So
    any resolution of ``woof.cycle`` is by definition a resolution
    against the live spine tree, which is the exact thing the port
    refuses -- only here it is refused early and legibly.

``woof`` itself is refused until the guard is ARMED with a pinned root.
    Arming is the worker saying "this checkout, and only this one".
    After arming, :func:`verify_pinned` re-reads
    ``sys.modules['woof'].__file__`` and confirms it really did land
    under the pinned root.  That check is recorded in the worker's
    receipt, so the evidence that the constraint held travels with the
    run instead of being asserted in prose.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
from typing import Any

#: The spine package prefix that may never be imported in a worker.
SPINE_PREFIX = "woof.cycle"
ROOT_PACKAGE = "woof"


class SpineImportRefused(ImportError):
    """A forecast process tried to import the cycling spine."""


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


class SpineImportGuard(MetaPathFinder):
    """Refuses spine imports inside a forecast process."""

    def __init__(self) -> None:
        self.pinned_root: Path | None = None
        self.armed = False

    # -- installation ------------------------------------------------------

    def install(self) -> "SpineImportGuard":
        already = sys.modules.get(ROOT_PACKAGE)
        if already is not None:
            raise SpineImportRefused(
                f"{ROOT_PACKAGE!r} is already imported at "
                f"{getattr(already, '__file__', '<unknown>')!r} before the "
                "guard was installed; a forecast worker must be entered with "
                "no woof in sys.modules so the port's pinned WOOF-ArWen checkout "
                "is the first and only one to claim the name")
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)
        return self

    def arm(self, pinned_root: str | Path) -> "SpineImportGuard":
        """Declare the one checkout ``woof`` is allowed to come from.

        Arming also PUTS that checkout on ``sys.path``, at the front.
        That is not a convenience: the port reaches for woof earlier
        than its own Arwen pin does.  ``KernelCache.__init__`` builds a
        compile manifest through
        ``woof.certify.compile_platform.compile_platform_fingerprint``
        -- the NVRTC numeric-identity fingerprint -- and that happens
        before ``_construct_device_stack`` ever calls
        ``pin_arwen_physics_v841``.  With no woof on the path at all the
        port refuses with ``cuda.compile_platform_fingerprint refused:
        ... No module named 'woof'``.

        So arming is the worker saying "the pinned checkout, from here
        on" and then making that true.  The insert is exactly the one
        ``_load_pinned_arwen_factory`` performs, only earlier, and the
        guard still refuses ``woof.cycle`` from anywhere -- including
        from this checkout, should a future Arwen ever grow one.
        """
        self.pinned_root = Path(pinned_root).expanduser().resolve()
        self.armed = True
        if str(self.pinned_root) not in sys.path:
            sys.path.insert(0, str(self.pinned_root))
        return self

    # -- the finder hook ---------------------------------------------------

    def find_spec(self, fullname: str, path: Any = None,
                  target: Any = None) -> None:
        if fullname == SPINE_PREFIX or fullname.startswith(SPINE_PREFIX + "."):
            raise SpineImportRefused(
                f"{fullname!r} is the cycling spine and must never be "
                "imported inside an MPAS forecast process. The port pins "
                "woof by commit and loads it from a frozen WOOF-ArWen checkout; "
                "importing the spine claims the package name first and the "
                "port then refuses with 'woof was already imported from a "
                "different tree'. Anything this worker needs from the spine "
                "belongs in mpas_cycle_bridge, which is gpuwm-free by "
                "design. The channel between the two processes is the "
                "anchor on disk, not a shared import.")
        if fullname == ROOT_PACKAGE or fullname.startswith(ROOT_PACKAGE + "."):
            if not self.armed:
                raise SpineImportRefused(
                    f"{fullname!r} was imported before the forecast worker "
                    "armed its import guard with the port's pinned WOOF-ArWen "
                    "checkout. Only the pinned checkout may claim the woof "
                    "name in this process; arm the guard first, or -- far "
                    "more likely -- this import does not belong here at all.")
        # Armed: fall through to the normal finders.  The port inserts its
        # pinned checkout at sys.path[0] itself, and verify_pinned() below
        # confirms after the fact that that is where woof came from.
        return None

    # -- the after-the-fact proof -----------------------------------------

    def verify_pinned(self) -> dict[str, Any]:
        """Report where ``woof`` actually came from, and whether it obeyed.

        Returns a receipt block rather than only raising, because the
        worker writes it into its segment receipt: the claim "this
        forecast used the pinned Arwen and nothing else" should be
        checkable from the evidence, not taken on trust.
        """
        module = sys.modules.get(ROOT_PACKAGE)
        if module is None:
            return {"gpuwm_imported": False, "pinned_root":
                    None if self.pinned_root is None else str(self.pinned_root),
                    "loaded_root": None, "obeyed": None,
                    "spine_modules": []}
        origin = getattr(module, "__file__", None)
        loaded_root = (None if origin is None
                       else Path(origin).resolve().parent.parent)
        obeyed = bool(self.pinned_root is not None and loaded_root is not None
                      and _is_under(loaded_root, self.pinned_root))
        spine = sorted(name for name in sys.modules
                       if name == SPINE_PREFIX
                       or name.startswith(SPINE_PREFIX + "."))
        receipt = {
            "gpuwm_imported": True,
            "pinned_root": (None if self.pinned_root is None
                            else str(self.pinned_root)),
            "loaded_root": None if loaded_root is None else str(loaded_root),
            "obeyed": obeyed,
            "spine_modules": spine,
        }
        if spine:  # pragma: no cover - the guard makes this unreachable
            raise SpineImportRefused(
                "the cycling spine is live inside a forecast process: "
                f"{spine}. The guard was bypassed.")
        if not obeyed:
            raise SpineImportRefused(
                f"woof was loaded from {receipt['loaded_root']!r}, not from "
                f"the pinned WOOF-ArWen checkout {receipt['pinned_root']!r}. The "
                "forecast in this process is not the pinned physics and its "
                "numbers must not be published.")
        return receipt


def install_guard() -> SpineImportGuard:
    """Install a fresh guard.  Call this first, before anything else."""
    return SpineImportGuard().install()
