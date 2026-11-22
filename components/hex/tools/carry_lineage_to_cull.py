#!/usr/bin/env python3
"""Give a culled child file the lineage its parent already had.

WHY THIS EXISTS, and the concrete breakage it removes.

``rw_mpas_mesh --cull-parent`` reproduces MPAS-Limited-Area v2.2 byte for
byte, and that tool writes exactly two global attributes into the regional
file -- ``on_a_sphere`` and ``sphere_radius``.  Everything else in the
parent's header is dropped.  That is correct and it is an anchor: the cull is
matching the native tool's recorded arrays on all three reference culls and nothing
here may move it.

``rw_mpas_lbc`` reads its child's header for twelve lineage attributes
(``model_name``, ``core_name``, ``version``, ``source``, ``Conventions``,
``git_version``, the five mesh-geometry attributes, and ``mesh_spec``) plus
``file_id``, and refuses by name rather than invent one -- *"inventing a value
here would stamp the stream with an identity its mesh never had"*.  A culled
child therefore cannot be handed to the boundary producer at all.

TWO THINGS ARE MISSING AND THEY ARE NOT THE SAME MISSING.

1. The attributes the parent HAD and the cull dropped.  Those are copied
   verbatim.  The child's lineage IS the parent's lineage: a cull moves no
   cell centre, invents no field and changes no configuration.
2. ~~Four attributes THE PARENT NEVER HAD.~~  **RETIRED 2026-08-26.**  This
   tool used to MINT ``model_name``, ``core_name``, ``version`` and
   ``git_version`` because no init this program produced carried them, which
   made every woof hex init unusable as a boundary producer's ``--grid``.
   The engine writes them now: ``rw_mpas_init`` takes the four as optional
   switches, defaults them to its own identity, and the init door passes this
   port's (``woof hex``, never ``mpas``).  A parent minted at or after that
   change carries all four and they are CARRIED like everything else.
   Minting survives only as a fallback for a parent minted BEFORE it, and
   the receipt still records which were carried and which were minted so a
   reader can tell an old parent from a new one at a glance.

Usage::

    python tools/carry_lineage_to_cull.py --parent PARENT.nc --child CULL.nc \
        --receipt lineage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: What ``rw_mpas_lbc``'s ``HeaderSource::from_grid`` requires, in the order
#: the native header emits them (rw-mpas/src/lbc/emit.rs:66-82).
REQUIRED = (
    "model_name",
    "core_name",
    "version",
    "source",
    "Conventions",
    "git_version",
    "on_a_sphere",
    "sphere_radius",
    "is_periodic",
    "x_period",
    "y_period",
    "mesh_spec",
    "file_id",
)


def port_identity() -> dict[str, str]:
    """The port's identity, as a FALLBACK for a parent minted before 2026-08-26.

    ``rw_mpas_init`` writes these four itself now (``init/emit.rs``, driven
    by ``init_door``'s ``--model-name``/``--core-name``/``--model-version``/
    ``--git-version``), so a parent this program minted at or after that
    change carries them and this mapping is never reached for it.  It stays
    for the archived parents that predate the fix, and it stays spelled the
    same way, so a cull of an old parent and a cull of a new one carry the
    same identity.

    ``model_name`` names THIS program.  Writing ``mpas`` here would be the
    identity the producer's refusal exists to prevent: these files carry
    WOOF-ArWen's physics and this port's numerics, and a stream stamped ``mpas``
    would claim a provenance no file in the chain has.
    """

    from woof.hex import __version__

    return {
        "model_name": "woof hex",
        "core_name": "atmosphere",
        "version": str(__version__),
        "git_version": f"gpuwm-hex-{__version__}",
    }


def carry(parent: Path, child: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    minted = port_identity()
    carried: dict[str, Any] = {}
    stamped: dict[str, Any] = {}
    with Dataset(str(parent), "r") as source:
        available = {name: source.getncattr(name) for name in source.ncattrs()}
    with Dataset(str(child), "a") as target:
        present = set(target.ncattrs())
        for name in REQUIRED:
            if name in present:
                continue
            if name in available:
                target.setncattr(name, available[name])
                carried[name] = _plain(available[name])
            elif name in minted:
                target.setncattr(name, minted[name])
                stamped[name] = minted[name]
            else:
                raise SystemExit(
                    f"{parent} carries no {name!r} and this tool mints only "
                    f"{sorted(minted)}; the boundary producer needs it and "
                    f"neither file can supply it"
                )
        # Every remaining parent attribute the cull dropped, carried too: a
        # child that answers "what configuration made me" with a shorter list
        # than its parent is a receipt with holes in it.
        for name, value in available.items():
            if name in present or name in carried or name in stamped:
                continue
            target.setncattr(name, value)
            carried[name] = _plain(value)
        final = sorted(target.ncattrs())
    return {
        "schema": "gpuwm-hex.cull-lineage/v1",
        "parent": str(parent),
        "child": str(child),
        "carried_from_parent": carried,
        "minted_from_port_identity": stamped,
        # Empty from 2026-08-26 onward for any parent rw_mpas_init wrote
        # after the lineage fix; a non-empty list here says the parent
        # predates it.
        "parent_predates_engine_lineage": sorted(stamped),
        "child_attributes_after": final,
    }


def _plain(value: Any) -> Any:
    try:
        return value.item()
    except AttributeError:
        return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    arguments = parser.parse_args(argv)
    report = carry(arguments.parent, arguments.child)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.receipt is not None:
        arguments.receipt.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
