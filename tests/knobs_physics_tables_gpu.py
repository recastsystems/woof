"""Real RUC GPU table edits at both endpoints, including unexposed land classes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from woof import physics_params as pp
    from woof.knobs import physics
    from woof.core.ruc import load_ruc_parameters
    from woof.core.physics_param_gpu import scale_physics_param_values
    bundle = load_ruc_parameters()
    result = []
    for row in physics.rows():
        if not row["parameter"].startswith("ruc."):
            continue
        for endpoint in ("min", "max"):
            pset = physics.make_set({row["name"]: row[endpoint]})
            tuned = pp.apply_ruc_edits(bundle, pset, scale_physics_param_values)
            spec = pp.registry()[row["parameter"]]
            changed = []
            for section, table in bundle.vegetation.items():
                after = tuned.vegetation[section]
                for index, (before_row, after_row) in enumerate(zip(table.rows, after.rows), 1):
                    before, current = asdict(before_row), asdict(after_row)
                    keys = [key for key in before if before[key] != current[key]]
                    selected = index in spec.categories.get(section, ())
                    assert keys == ([spec.column] if selected else []), (row["name"], section, index, keys)
                    if selected:
                        if spec.column == "z0":
                            assert before["ifor"] != 7
                            assert current["z0"] > 0
                        changed.append({"section": section, "category": index, "column": spec.column,
                                        "before": before[spec.column], "after": current[spec.column]})
            assert changed
            result.append({"name": row["name"], "endpoint": endpoint,
                "changed_cells": changed, "sha256": hashlib.sha256(
                    json.dumps(changed, sort_keys=True).encode()).hexdigest()})
    args.out.write_text(json.dumps({"status": "PASS", "rows": result}, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
