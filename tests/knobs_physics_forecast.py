"""Real prepared-state forecast receipt for one physics-knob endpoint.

Run only inside the card queue. Inputs are a shared WRF real directory;
outputs are disposable lane-local files. Hashes cover every written variable
through the Rust NetCDF reader, not NetCDF file metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def profile_text(text, profile):
    """Set explicit test selectors even when the importer omitted defaults."""
    import re
    overrides = {}
    if profile == "length2":
        overrides = {"bl_mynn_mixlength": 2, "bl_mynn_version": "gsd_41"}
    elif profile in ("deep", "shallow"):
        overrides = {"cu_physics": 3, "cudt_minutes": 0}
        if profile == "shallow":
            overrides.update(ishallow=1)
    elif profile == "radius":
        overrides = {"use_mp_re": 1}
    for key, value in overrides.items():
        encoded = json.dumps(value)
        line = key + " = " + encoded
        text, count = re.subn(rf"(?m)^{key}\s*=.*$", line, text)
        if count == 0:
            if text.count("[shared]\n") != 1:
                raise ValueError("profile needs exactly one shared run table")
            text = text.replace("[shared]\n", "[shared]\n" + line + "\n", 1)
        elif count != 1:
            raise ValueError("profile needs one explicit " + key + " selector")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--prepared-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--knob")
    parser.add_argument("--value", type=float, default=0)
    parser.add_argument("--all-zero", action="store_true")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--profile", choices=("base", "length2", "deep", "shallow", "radius"), default="base")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from woof.knobs import physics
    values = ({row["name"]: 0 for row in physics.rows()} if args.all_zero else
              ({args.knob: args.value} if args.knob else {}))
    if args.prepared_root:
        env = physics.write_runtime(values, args.out / "parameters.toml")
        os.environ.update(env)
        pset = physics.activate(values)
        inputs = prepared_inputs(args.prepared_root, args.seconds, args.profile)
        from woof import physics_params as pp
        if pset:
            pp.check_schemes(pset, [d.run for d in inputs.experiment.domains], source="prepared trial")
        inputs = replace(inputs, experiment=replace(inputs.experiment, physics_params=pset))
        from woof.prepared_single_domain_forecast import run_prepared_forecast
        start = time.monotonic()
        (args.out / "forecast").mkdir()
        result = run_prepared_forecast(inputs, output_directory=args.out / "forecast")
        identity = {"prepared_root": str(args.prepared_root), "files": dict(inputs.file_sha256),
                    "derived_output_cadence_seconds": args.seconds,
                    "rule": "sealed inputs and physics; explicit short output-only trial"}
    else:
        inputs, result, start = wrf_forecast(args, values)
        identity = dict(inputs.source_identity)
    from woof import netcdf_bridge as nc
    frames = {}
    for path in sorted((args.out / "forecast").rglob("wrfout_d??*")):
        if not path.is_file() or path.suffix == ".json":
            continue
        with nc.Dataset(path) as dataset:
            record = {}
            for name, variable in sorted(dataset.variables.items()):
                array = np.asarray(variable[:])
                record[name] = {"dtype": array.dtype.str, "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True}
            frames[path.name] = record
    if not frames or not all(item["finite"] for frame in frames.values() for item in frame.values()):
        raise RuntimeError("forecast lacks finite full-field output")
    receipt = {"knobs": values, "seconds": args.seconds, "wall_seconds": time.monotonic()-start,
        "profile": args.profile, "strict": os.environ.get("GPUWM_WRF_EXACT") == "1",
        "frames": frames, "source_identity": identity, "result": result}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str)+"\n")
    deleted = []
    for path in sorted((args.out / "forecast").rglob("wrfout_d??*")):
        if path.is_file() and path.suffix != ".json":
            if not path.resolve().is_relative_to(args.out.resolve()):
                raise ValueError("raw output deletion target escaped the owned run directory")
            deleted.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()
    (args.out / "deleted-data.json").write_text(json.dumps(deleted, indent=2)+"\n")
    return 0


def prepared_inputs(root, seconds, profile):
    """Reuse the sealed analysis; never substitute preparation physics."""
    from woof.stage_cli import resolve_bundle, single_domain_digests
    from woof.prepared_single_domain_forecast import preflight_prepared_forecast, _clock_defaults
    bundle = resolve_bundle(root)
    if bundle["layout"] != "single":
        raise ValueError("this verification driver requires a single prepared domain")
    declared = bundle["payload"]["execution_inputs"]
    def authority(key, filename):
        p = root / filename
        if not p.is_file():
            spec = declared[key]
            p = Path(spec["path"])
            if hashlib.sha256(p.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("prepared authority bytes changed")
        return p
    digests = single_domain_digests(bundle)
    config = authority("experiment_config", "experiment.toml")
    _, prepared_cadence = _clock_defaults(config)
    inputs = preflight_prepared_forecast(source=bundle["source"], prepared_root=root,
        proof_sha256=digests["proof"], source_manifest_sha256=digests["source_manifest"],
        prepared_content_sha256=digests["prepared_content"],
        experiment_config=config,
        wps_namelist=authority("wps_namelist", "namelist.wps"),
        run_seconds=seconds, history_interval_seconds=prepared_cadence, devices=1)
    cfg = inputs.experiment.root.run
    if profile == "length2" and (cfg.bl_mynn_mixlength != 2 or cfg.bl_mynn_version != "gsd_41"):
        raise ValueError("prepared suite is not the length2 consumer; no physics substitution")
    if profile == "radius" and cfg.use_mp_re != 1:
        raise ValueError("prepared suite does not use microphysics radii; no physics substitution")
    if profile not in ("base", "length2", "radius"):
        raise ValueError("prepared driver preserves its cumulus and microphysics selectors")
    domains = tuple(replace(d, history_interval_s=seconds, history_begin_s=0,
                           history_end_s=seconds, run=replace(d.run, output_interval_s=seconds))
                    for d in inputs.experiment.domains)
    return replace(inputs, experiment=replace(inputs.experiment, run_seconds=seconds, domains=domains))


def wrf_forecast(args, values):
    from woof.knobs import physics
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.wrfinput_forecast import prepare_wrf_run, WrfInitialization
    from woof.prepared_domain_tree_forecast import run_prepared_tree
    from woof.progress_log import ProgressOptions
    run = resolve_wrfinput_run(args.input, rrtmg_variant="rrtmg_legacy")
    # Keep only the initial and final field records needed for this check.
    import re
    text, count = re.subn(r"(?m)^history_interval_s\s*=.*$",
                         f"history_interval_s = {args.seconds}.0", run.toml_text)
    if count != 1:
        raise ValueError("resolved source must contain one history interval")
    # A length2 knob needs that admitted arm; no case-specific core default.
    text = profile_text(text, args.profile)
    run = replace(run, toml_text=text)
    env = physics.write_runtime(values, args.out / "parameters.toml")
    os.environ.update(env)
    physics.activate(values)
    start = time.monotonic()
    inputs = prepare_wrf_run(run, args.out / "prepared", run_seconds=args.seconds)
    (args.out / "forecast").mkdir()
    result = run_prepared_tree(inputs, output_directory=args.out / "forecast",
        io_mode="history", initialization=WrfInitialization(inputs),
        progress_options=ProgressOptions(progress_every=150))
    return inputs, result, start


if __name__ == "__main__":
    raise SystemExit(main())
