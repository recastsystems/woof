"""Verify SST landings and hash actual Rust-written TMN forecast frames."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

def valid_time(path):
    text = path.name.removeprefix("wrfout_d01_")
    for form in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H_%M_%S"):
        try:
            return datetime.strptime(text, form)
        except ValueError:
            pass
    raise ValueError(f"unrecognized history valid time: {path.name}")

def history_paths(root):
    """Native history datasets, excluding their JSON authority sidecars."""
    return sorted((path for path in Path(root).rglob("wrfout_d01_*")
                   if path.is_file() and path.suffix != ".json"), key=valid_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dt", type=float, required=True)
    args = parser.parse_args()
    from woof.netcdf_bridge import open_dataset
    from woof.surface_analysis import selected_records, _instant, _read_plane
    candidates = []
    for path in args.forecast.rglob("manifest.json"):
        doc = json.loads(path.read_text())
        if doc.get("schema") == "gpuwm-surface-analysis-v1":
            candidates.append((path, doc))
    if len(candidates) != 1:
        raise ValueError("forecast must contain exactly one prepared surface manifest")
    manifest, document = candidates[0]
    from woof.experiment import load_experiment
    experiment = load_experiment(manifest.parent.parent/"experiment.toml")
    cfg = experiment.root.run
    if (cfg.tmn_update,cfg.lagday,cfg.sst_update)!=(1,150,1) or float(cfg.dt)!=args.dt:
        raise ValueError("prepared configuration does not select TMN/SST updating with the claimed lag and time step")
    paths = history_paths(args.forecast)
    if not paths:
        raise ValueError("forecast has no actual history frames")
    start = valid_time(paths[0])
    if start != experiment.start_time.replace(tzinfo=None):
        raise ValueError("first history time does not match the prepared model origin")
    origin = _instant(start.isoformat())
    frames = []; consumed = set()
    for path in paths:
        elapsed = (valid_time(path)-start).total_seconds()
        numeric = {}; required = {}; checks = []
        with open_dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if np.dtype(variable.dtype).kind not in "fiu":
                    continue
                values = np.asarray(variable.read_transformed(cache=False))
                numeric[name] = dict(stored_dtype=str(variable.dtype), decoded_dtype=str(values.dtype),
                    shape=list(values.shape), sha256=hashlib.sha256(values.tobytes()).hexdigest())
                if name in ("TMN", "TSK", "T2", "SST"):
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite {name} in {path.name}")
                    required[name] = values
            if set(required) != {"TMN", "TSK", "T2", "SST"}:
                raise ValueError(f"missing acceptance fields in {path.name}: {set(required)}")
        for row in selected_records(document["records"], origin+max(0,elapsed-args.dt)):
            if row["target"] != "SST":
                continue
            expected = _read_plane(manifest.parent,row,document["shape"])
            valid = np.isfinite(expected)
            actual = np.asarray(required["SST"],dtype="<f4").reshape(expected.shape)
            if not valid.any() or actual[valid].tobytes() != expected[valid].tobytes():
                raise ValueError(f"SST source words differ in {path.name} at {row['valid_time']}")
            consumed.add(row["valid_time"])
            checks.append(dict(valid_time=row["valid_time"], valid_cells=int(valid.sum()), source_words_match=True))
        stats = {name:dict(min=float(value.min()), max=float(value.max()), mean=float(value.mean()))
                 for name,value in required.items()}
        frames.append(dict(file=path.name, elapsed_seconds=elapsed,
            sha256_file=hashlib.sha256(path.read_bytes()).hexdigest(),fields=numeric,statistics=stats,SST=checks))
    if frames[-1]["elapsed_seconds"] <= 86400 or len(consumed)<2:
        raise ValueError("acceptance needs a multi-day forecast and at least two consumed SST times")
    args.output.write_text(json.dumps(dict(status="PASS",tmn_update=1,lagday=150,sst_update=1,
        dt=args.dt,completed_seconds=frames[-1]["elapsed_seconds"],frames=frames,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        SST_valid_times_checked=sorted(consumed),hash_encoding="Rust-decoded numeric arrays, generally float64; not resident-state hashes or a WRF forecast comparison"),indent=2))

if __name__ == "__main__":
    main()
