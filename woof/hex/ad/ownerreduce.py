# SPDX-License-Identifier: Apache-2.0
"""The owner-reduced kernel set a rank plan needs: a row's kernel set with the selected divergence-damping row added.

A rank plan runs divergence damping as two selected passes (interior, then the boundary after the halo receive), and
its adjoint is the owner-reduced image (engine commit 51d549ac): the boundary pass gathers every edge
of the phase, one float64 sum a cell in incidence order, as the one-card ``divergence_damping_f32__ad`` does. A
one-card kernel set has no such row, and a rank window on it stops at its first selected launch
(``DEFECT: divergence_damping_selected_f32: required damping derivative unavailable: no derivative row``; storm box,
two ranks, 2026-11-13).

``build`` is the box chain's ``add_selected_row.py BASE OUT`` (tree/ad/phys/multirank/dd_selected/ownerreduce) with
the selected-row folder as an argument instead of a box path: the base set's rows unchanged, its images linked, the
selected row taken from the folder (replacing a base row of that name), its spec and image copied in. ``ensure``
builds it once per (base set, selected row) under the row cache, so the door never hand-assembles a set.

Selected-row folder layout (one download member, ``ranks/ownerreduce``): ``kernels.json`` (holding the row),
``specs/divergence_damping_selected_f32.json``, ``cubins/divergence_damping_selected_f32.fatbin`` and
``cubins/divergence_damping_selected_f32.provenance.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

FUNCTION = "divergence_damping_selected_f32"


class OwnerReduceError(Exception):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(base: Path, selected: Path, out: Path, cubins: Path | None = None) -> dict:
    """add_selected_row.py BASE OUT, with the selected row read from ``selected``. ``cubins``: the base set's image
    folder when it is not BASE/cubins (a row names its kernels and cubins apart). Returns the overlay record."""
    base, selected, out = Path(base), Path(selected), Path(out)
    src_dirs = {"cubins": Path(cubins) if cubins is not None else base / "cubins", "specs": base / "specs"}
    F = FUNCTION
    for need in (base / "kernels.json", selected / "kernels.json", selected / "specs" / f"{F}.json",
                 selected / "cubins" / f"{F}.fatbin", selected / "cubins" / f"{F}.provenance.json"):
        if not need.is_file():
            raise OwnerReduceError(f"the owner-reduced kernel set needs {need}, which is missing")
    os.makedirs(out / "cubins", exist_ok=True)
    os.makedirs(out / "specs", exist_ok=True)
    k = json.loads((base / "kernels.json").read_text(encoding="utf-8"))
    rows = [r for r in json.loads((selected / "kernels.json").read_text(encoding="utf-8"))["kernels"]
            if r.get("function") == F]
    if len(rows) != 1:
        raise OwnerReduceError(f"{selected / 'kernels.json'} holds {len(rows)} rows for {F}, not 1")
    row = rows[0]
    sp = os.path.abspath(out / "specs" / f"{F}.json")
    shutil.copyfile(selected / "specs" / f"{F}.json", sp)
    row["derivative"]["spec"] = {"path": sp, "sha256": _sha(Path(sp))}
    k["kernels"] = [r for r in k["kernels"] if r.get("function") != F] + [row]
    k["overlay_ownerreduce"] = {"base": str(base), "added_or_replaced": F,
                                "fatbin_sha256": _sha(selected / "cubins" / f"{F}.fatbin")}
    (out / "kernels.json").write_text(json.dumps(k, indent=1), encoding="utf-8")
    for d, src in src_dirs.items():
        if not src.is_dir():
            continue
        for f in os.listdir(src):
            p = out / d / f
            if f.startswith(F) or os.path.lexists(p):
                continue
            os.symlink(os.path.realpath(src / f), p)
    for f in (f"{F}.fatbin", f"{F}.provenance.json"):
        p = out / "cubins" / f
        if os.path.lexists(p):
            os.remove(p)
        shutil.copyfile(selected / "cubins" / f, p)
    for x in os.listdir(base):
        if x not in ("kernels.json", "cubins", "specs") and not os.path.lexists(out / x):
            os.symlink(os.path.realpath(base / x), out / x)
    return k["overlay_ownerreduce"]


def ensure(base_kernels: Path, selected: Path, cache: Path, cubins: Path | None = None) -> dict:
    """{"kernels": ..., "cubins": ...} of the owner-reduced set over the kernel set of ``base_kernels``, built once
    under ``cache/kset-own/<id>`` (the id is the base kernels.json and the selected row's files)."""
    base = Path(base_kernels).parent
    F = FUNCTION
    h = hashlib.sha256()
    for p in (Path(base_kernels), Path(selected) / "kernels.json", Path(selected) / "cubins" / f"{F}.fatbin",
              Path(selected) / "specs" / f"{F}.json"):
        if not p.is_file():
            raise OwnerReduceError(f"the owner-reduced kernel set needs {p}, which is missing")
        h.update(_sha(p).encode())
    h.update(str(Path(cubins).resolve() if cubins is not None else base.resolve() / "cubins").encode())
    out = Path(cache) / "kset-own" / h.hexdigest()[:16]
    done = out / ".complete.json"
    if not done.is_file():
        tmp = out.with_name(out.name + f".tmp{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        record = build(base, selected, tmp, cubins)
        # the spec path inside kernels.json names the folder it was built in: rename it to the final folder
        kj = tmp / "kernels.json"
        kj.write_text(kj.read_text(encoding="utf-8").replace(str(os.path.abspath(tmp)), str(os.path.abspath(out))),
                      encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)
        (out / ".complete.json").write_text(json.dumps({**record, "base_kernels": str(base_kernels),
                                                        "selected": str(selected)}, indent=1) + "\n")
    return {"kernels": out / "kernels.json", "cubins": out / "cubins"}
