#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the recorder's table of kernel kinds whose derivative this release has checked (data/derivative-kinds.v1.json).

    derivative_kinds.py OUT.json PLAN.tar.gz|PLAN_DIR ...

A kind is listed when it runs in a plan whose window passed the release's own check: the example plans the release
pins (data/ad-assets.json, "ad-plan-*"). The recorder reads the table when it seals a plan of its own (coverage.py):
a kind that is in no such plan, has no rule by name in the window and is not held by name refuses the recording.
Each kind also lists the pairs of its pointer arguments (by argument index) that shared one address in some launch of
those plans ("aliased_args"): a launch in a plan of one's own whose arguments share an address in any other pair runs
the kind in a form its derivative was not checked in, and refuses too (coverage.aliased_pairs says why).
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def plan_of(source: Path) -> tuple[dict, str]:
    if source.is_dir():
        return json.loads((source / "plan.json").read_text()), source.name
    with tarfile.open(source) as archive:
        member = next(m for m in archive.getmembers() if m.name == "plan.json" or m.name.endswith("/plan.json"))
        return json.load(archive.extractfile(member)), source.name


def aliased_pairs(op: dict) -> set[tuple[int, int]]:
    """The pairs (i, j), i < j, of one launch's pointer arguments that name the same address (allocation, offset)."""
    seen: dict[tuple, list[int]] = {}
    for i, arg in enumerate(op.get("arguments") or ()):
        target = arg.get("target") if isinstance(arg, dict) else None
        if isinstance(target, dict) and "allocation" in target:
            seen.setdefault((target["allocation"], target.get("offset", 0)), []).append(i)
    return {(a[x], a[y]) for a in seen.values() for x in range(len(a)) for y in range(x + 1, len(a))}


def main(argv: list[str]) -> int:
    out, sources = Path(argv[0]), [Path(a) for a in argv[1:]]
    kinds: dict[str, dict] = {}
    plans = []
    for source in sources:
        plan, name = plan_of(source)
        count = collections.Counter(op["function"] for ops in plan["blocks"].values() for op in ops
                                    if op.get("kind") == "kernel")
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
        plans.append({"plan": name, "kinds": len(count), "launches": sum(count.values()), "plan_json_digest": digest})
        for kind in count:
            kinds.setdefault(kind, {"plans": []})["plans"].append(name)
        for op in (op for ops in plan["blocks"].values() for op in ops if op.get("kind") == "kernel"):
            pairs = aliased_pairs(op)
            if pairs:
                have = {tuple(p) for p in kinds[op["function"]].get("aliased_args", [])}
                kinds[op["function"]]["aliased_args"] = sorted(map(list, have | pairs))
    doc = {"schema": "hexad.derivative-kinds/v1",
           "what": "kernel kinds that run in an example plan whose window passed this release's check; the recorder "
                   "refuses a plan of its own that runs a kind outside this table, the window's rules by name and "
                   "the rows' named holds (woof.hex.ad.record.coverage); aliased_args: the pointer-argument pairs of the "
                   "kind that shared one address in some launch of those plans, the only aliasing a plan of one's own "
                   "may run the kind with",
           "plans": plans, "kinds": dict(sorted(kinds.items()))}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(kinds)} kinds from {len(plans)} plans -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
