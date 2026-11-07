"""Post-run GPU family products: python -m woof.ensemble.family_products RUN --out OUT.

RUN may be an ensemble-set.json, an ensemble-manifest.json, or their folder.
An explicit JSON member map supplies families for older receipts. Its shape is
{"0": {"family": "ai", "family_weight": 0.5}, "1": {...}}.
No model identity is inferred from a file name or a source spelling.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import time

import numpy as np

from woof.ensemble.batch_products import (FieldProducts, apply_member_weights,
    default_product_requests, prepare_product_frame, write_product_frame)

SCHEMA = "gpuwm-family-products.v1"


def family_roster(members, *, member_map=None):
    """Resolve complete lineage and divide each family's recorded mass.

    Existing member weights retain their relative probabilities *within* a
    family. Family mass is normalized across the represented families once.
    """
    rows = []
    supplied = {str(k): v for k, v in (member_map or {}).items()}
    for member in members:
        identity = member.get("member_id", member.get("member"))
        provenance = member.get("provenance", member)
        overrides = supplied.get(str(identity), {})
        if isinstance(overrides, str):
            overrides = {"family": overrides}
        family = overrides.get("family", provenance.get("parent_family", provenance.get("family", member.get("parent_family", member.get("family")))))
        weight = overrides.get("family_weight", provenance.get("family_weight", member.get("family_weight")))
        if not isinstance(family, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", family) or family in ("combined", ".", ".."):
            raise ValueError(f"member {identity} needs an explicit safe family identity in lineage or member map")
        rows.append(dict(member_id=identity, family=family, family_weight=weight, provenance=provenance,
                         family_origin="member-map" if overrides else "lineage",
                         within_weight=float(member.get("weight", 1.))))
    ids = [row["member_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids) or None in ids:
        raise ValueError("family products need a complete distinct member roster")
    if set(supplied) - {str(identity) for identity in ids}:
        raise ValueError("member map contains IDs outside the recorded roster")
    families = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    masses = {}
    recorded = [row["family_weight"] is not None for row in rows]
    if any(recorded) and not all(recorded):
        raise ValueError("family weights must be recorded for every member or default for all")
    for name, group in families.items():
        values = {float(row["family_weight"]) for row in group} if all(recorded) else {1.}
        if len(values) != 1:
            raise ValueError(f"inconsistent recorded family weight for {name}")
        masses[name] = values.pop()
    if any(not math.isfinite(v) or v < 0 for v in masses.values()) or not any(masses.values()):
        raise ValueError("family weights must be finite nonnegative masses with a positive total")
    total = math.fsum(masses.values())
    if not math.isfinite(total):
        raise ValueError("family weight total must be finite")
    for name in masses:
        masses[name] /= total
    for name, group in families.items():
        if any(not math.isfinite(row["within_weight"]) or row["within_weight"] < 0 for row in group):
            raise ValueError("member probability weights must be finite nonnegative masses")
        total = math.fsum(row["within_weight"] for row in group)
        if not math.isfinite(total) or (total == 0 and masses[name] > 0):
            raise ValueError("represented family needs finite positive member probability mass")
        for row in group:
            within = row.pop("within_weight")
            row["conditional_weight"] = within / total if total else 1. / len(group)
            row["family_weight"] = masses[name]
            row["weight"] = row["conditional_weight"] * masses[name]
    return {"members": rows, "family_weights": masses,
            "weight_origin": ("member-map" if any(isinstance(value, dict) and "family_weight" in value
                                                 for value in supplied.values()) else "recorded")
                              if all(recorded) else "equal-family-default"}


def reduce_family_frame(fields, requests, roster, *, available_bytes, array_module=None):
    """Yield one family or combined aggregate at a time, using shared reducers.

    The caller writes each product before advancing. Only one group's output
    and one field stack need reside on the card at once.
    """
    if array_module is None:
        import cupy as array_module
    xp = array_module
    members = roster["members"]
    groups = [(name, [i for i, row in enumerate(members) if row["family"] == name])
              for name in sorted(roster["family_weights"])]
    groups.append(("combined", list(range(len(members)))))
    for name, indices in groups:
        selected = {field: xp.ascontiguousarray(values[indices]) for field, values in fields.items()}
        if any(values.shape[0] != len(members) for values in fields.values()):
            raise ValueError("diagnostic stack differs from the recorded family roster")
        products = prepare_product_frame(selected, requests, available_bytes=available_bytes,
                                         array_module=xp)
        products()
        weights = [members[i]["weight" if name == "combined" else "conditional_weight"] for i in indices]
        apply_member_weights(products, weights)
        attrs = {"family_product_schema": SCHEMA, "ensemble_family": name,
                 "family_weights": json.dumps(roster["family_weights"], sort_keys=True),
                 "family_weight_origin": roster["weight_origin"],
                 "ensemble_member_order": json.dumps([members[i]["member_id"] for i in indices]),
                 "member_probability_weights": json.dumps(weights),
                 "spread_definition": "weighted reliability sample variance; denominator 1-sum(normalized_weight^2)",
                 "ensemble_member_metadata": json.dumps([members[i] for i in indices]),
                 "pmm_distribution": "member probability mass per finite cell; descending empirical CDF; equal-rank draws averaged"}
        yield name, products, attrs
        del products, selected


def score_reflectivity(forecasts, observed, observed_valid, *, dx_m,
                       thresholds=(30., 40.), half_widths=(0, 1, 3, 5, 10)):
    """Use the existing Rust masked FSS and contingency kernels.

    A common mask across all products makes each family and combined row
    answer for exactly the same observed cells. Mean and PMM are labelled.
    """
    from woof.verify.obs.fss import masked_fss, box_length_m
    from woof.verify.obs.contingency import score_field
    valid = np.asarray(observed_valid, bool) & np.isfinite(observed)
    for values in forecasts.values():
        valid &= np.isfinite(values)
    rows = []
    for (family, statistic), values in forecasts.items():
        for threshold in thresholds:
            if not valid.any():
                for width in half_widths:
                    rows.append(dict(family=family, statistic=statistic, threshold=float(threshold),
                        hits=0, misses=0, false_alarms=0, correct_negatives=0, total=0,
                        observed_event_fraction=None, forecast_event_fraction=None,
                        probability_of_detection=None, false_alarm_ratio=None,
                        critical_success_index=None, frequency_bias=None,
                        equitable_threat_score=None, heidke_skill_score=None,
                        fss=None, threshold_model=float(threshold), threshold_obs=float(threshold),
                        half_width_cells=width, observed_base_rate=None, model_base_rate=None,
                        fss_useful=None, scored_cells=0, frequency_matched=False,
                        box_length_m=box_length_m(width, dx_m), score_status="no-common-valid-cells"))
                continue
            table = score_field(observed, values, threshold=threshold, valid=valid)
            for width in half_widths:
                row = dict(family=family, statistic=statistic, **table,
                           **masked_fss(values, observed, valid=valid, threshold=threshold,
                                        half_width=width, boundary="zero").record())
                row["box_length_m"] = box_length_m(width, dx_m)
                row["score_status"] = "scored" if row["total"] else "no-common-valid-cells"
                row = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                       for key, value in row.items()}
                rows.append(row)
    return rows


def _path(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    # Native receipts use output-root paths; mass receipts sit at that root.
    candidates = (root / path, root.parent / path)
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _clock(value):
    value = value[:10] + "T" + value[11:].replace("_", ":")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _read_history(path):
    """Rust reads stored diagnostic words; composition stays on the GPU."""
    from woof.netcdf_bridge import open_dataset
    with open_dataset(path) as dataset:
        names = dataset.variables
        result = {}
        for field, candidates in {"refl": ("REFD_MAX", "REFL_10CM"),
                "uh": ("UH", "UP_HELI_MAX"), "uh01": ("UP_HELI_MAX01",), "uh02": ("UP_HELI_MAX02",),
                "uh03": ("UP_HELI_MAX03",), "rain_total": ("RAINNC",)}.items():
            for name in candidates:
                if name in names:
                    values = np.asarray(names[name][:], np.float32)
                    if values.shape[0] == 1 and values.ndim > 2:
                        values = values[0]
                    result[field] = values
                    break
        for name in ("RAINC", "RAINSH"):
            if name in names and "rain_total" in result:
                result[name] = np.asarray(names[name][:], np.float32).squeeze(axis=0)
        result["latitude"] = np.asarray(names["XLAT"][:], np.float32).reshape(result["refl"].shape[-2:])
        result["longitude"] = np.asarray(names["XLONG"][:], np.float32).reshape(result["refl"].shape[-2:])
    return result


def _input_frames(path):
    """Read receipts without mutating the original run or its manifests."""
    if path.is_dir():
        if (path / "ensemble-set.json").is_file():
            path = path / "ensemble-set.json"
        else:
            # A run keeps resume generations under hidden folders
            # (.ensemble-resume/generations/<id>/d01/...); they are copies,
            # not more domains. Breakage it prevents: a one-domain run was
            # refused "name one ensemble manifest explicitly".
            manifests = sorted(found for found in path.glob("**/ensemble-manifest.json")
                               if not any(part.startswith(".") for part in found.relative_to(path).parts))
            if len(manifests) != 1:
                raise ValueError("name one ensemble manifest explicitly when a run contains multiple domains")
            path = manifests[0]
    receipt = json.loads(path.read_text())
    root = path.parent
    frames = {}
    if "completed" in receipt:
        members = sorted(receipt["completed"], key=lambda m: m["member_id"])
        for member in members:
            for frame in member["frames"]:
                key = (f'd{frame["grid_id"]:02d}-episode-{frame["episode"]:03d}', _clock(frame["valid_time"]))
                rows = frames.setdefault(key, {})
                if member["member_id"] in rows:
                    raise ValueError("duplicate member at one valid time")
                rows[member["member_id"]] = ("npz", _path(root, frame["path"]))
    else:
        members = receipt.get("member_metadata", [])
        if [m.get("member_id", m.get("member")) for m in members] != receipt.get("member_order"):
            raise ValueError("member metadata must retain manifest member order")
        for row in receipt.get("member_files", []):
            match = re.search(r"(\d{4}-\d\d-\d\d[_T]\d\d[:_]\d\d[:_]\d\d)", Path(row["path"]).name)
            if not match:
                raise ValueError("native history receipt has no valid time")
            key = (row.get("domain", path.parent.name), _clock(match[1]))
            rows = frames.setdefault(key, {})
            if row["member_id"] in rows:
                raise ValueError("duplicate native member frame")
            rows[row["member_id"]] = ("history", _path(root, row["path"]))
        # Histories can retain the accumulation baseline while later hours live in
        # deferred packs. Prefer a history at an overlap, filling the remaining
        # members and clocks from packs without weakening their identity checks.
        diagnostic_members = set()
        for frame in receipt.get("frames", []):
            key = (frame.get("domain", path.parent.name), _clock(frame["valid_time"]))
            for pack in frame.get("diagnostic_files", frame.get("packs", [])):
                rows = frames.setdefault(key, {})
                missing = []
                for position, identity in enumerate(pack["member_ids"]):
                    diagnostic_key = (*key, identity)
                    if diagnostic_key in diagnostic_members:
                        raise ValueError("duplicate diagnostic member frame")
                    diagnostic_members.add(diagnostic_key)
                    if identity not in rows:
                        missing.append((position, identity))
                if missing:
                    from woof.ensemble.product_worker import _identity
                    pack_path = _path(root, pack["path"])
                    _identity(pack_path, expected=pack)
                    for position, identity in missing:
                        rows[identity] = ("pack", (pack_path, position))
    if not frames:
        raise ValueError("no retained member diagnostics or histories; aggregate products cannot recover family distributions")
    return path, receipt, members, frames


def _read_input(item):
    kind, value = item
    if kind == "history":
        return _read_history(value)
    if kind == "npz":
        with np.load(value, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    from woof.netcdf_bridge import open_dataset
    path, position = value
    with open_dataset(path) as data:
        return {name: np.asarray(variable[position], np.float32)
                for name, variable in data.variables.items()
                if len(variable.shape) == 3}


def run_products(run, out, *, member_map=None, dx_m=3000., offline=False,
                 mrms_cache=None, renderer=None, hours=None, half_widths=(0, 1, 3, 5, 10)):
    import cupy as cp
    from woof import rustwx
    from woof.obs.mrms_fetch import MrmsCompositeCache, bbox_around
    from woof.obs.obspack import read_geo_pack, read_grid_pack
    from woof.verify.obs.regrid import build_plan, apply_plan
    from woof.ingest.cpu_backend import available_cpu_count
    source, receipt, members, frames = _input_frames(Path(run))
    roster = family_roster(members, member_map=member_map)
    order = [m["member_id"] for m in roster["members"]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = rustwx.find_ensemble_product_renderer(renderer)
    if renderer is None:
        raise RuntimeError("family products require the installed Rust ensemble renderer")
    result = dict(schema=SCHEMA, input_receipt=str(source.resolve()), roster=roster,
                  frames=[], scores=[], parallelism={"reducers": "GPU", "readers": available_cpu_count(),
                  "render": "Rust Rayon", "verification": "Rust Rayon"})
    caches = {}
    rain_origin = {}
    with ThreadPoolExecutor(max_workers=available_cpu_count()) as readers:
        # History precipitation accumulates from the run's earliest retained
        # instant even when a later hour alone is requested for rendering.
        for domain in sorted({key[0] for key in frames}):
            first_key = min(key for key in frames if key[0] == domain)
            first = frames[first_key]
            history_ids = [identity for identity in order if identity in first and first[identity][0] == "history"]
            for identity, array in zip(history_ids, readers.map(_read_input, [first[i] for i in history_ids])):
                if "rain_total" in array:
                    baseline = cp.asarray(array["rain_total"], dtype=cp.float32)
                    for key in ("RAINC", "RAINSH"):
                        if key in array:
                            baseline = baseline + cp.asarray(array[key])
                    rain_origin[(domain, identity)] = (baseline, first_key[1])
        for (domain, valid), rows in sorted(frames.items()):
            if hours is not None and valid not in hours:
                continue
            if hours is None and valid[14:] != "00:00":
                continue
            if set(rows) != set(order):
                raise ValueError(f"{domain} {valid}: missing members; partial-roster probabilities would change the denominator")
            started = time.perf_counter()
            arrays = list(readers.map(_read_input, [rows[identity] for identity in order]))
            if "latitude" in arrays[0]:
                lat, lon = arrays[0]["latitude"], arrays[0]["longitude"]
            else:
                from woof.netcdf_bridge import open_dataset
                from woof.ensemble.product_worker import _identity
                coordinate_record = receipt["coordinates_file"]
                coordinate = _path(source.parent, coordinate_record["path"])
                _identity(coordinate, expected=coordinate_record)
                with open_dataset(coordinate) as data:
                    lat, lon = (np.asarray(data.variables[key][:]) for key in ("XLAT", "XLONG"))
            if any("latitude" in a and (not np.array_equal(a["latitude"], lat) or not np.array_equal(a["longitude"], lon)) for a in arrays):
                raise ValueError("family members do not share one diagnostic grid")
            bbox = bbox_around(lat, lon)
            if bbox not in caches:
                cache_root = Path(mrms_cache or out / "mrms-cache")
                legacy_index = cache_root / "index.json"
                # Crop identity stays stable when selecting only later hours
                # or a different subset of grids. Accept older single-crop
                # roots only when their recorded extent matches this grid.
                if not (legacy_index.is_file() and json.loads(legacy_index.read_text()).get("bbox") == bbox):
                    cache_root /= hashlib.sha256(bbox.encode()).hexdigest()[:16]
                caches[bbox] = MrmsCompositeCache(cache_root, bbox=bbox, offline=offline)
            cache = caches[bbox]
            requests, missing = default_product_requests(set.intersection(*(set(a) for a in arrays)), postage_stamp=False)
            if not any(request.field == "refl" for request in requests):
                raise ValueError("family MRMS comparison requires retained composite or volume reflectivity")
            requests = tuple(replace(r, paintball=False, spaghetti=False,
                thresholds=tuple(sorted(set(r.thresholds) | {30., 40.}))) if r.field == "refl"
                else replace(r, paintball=False, spaghetti=False) for r in requests)
            forecasts, paths, maps = {}, {}, {}
            for request in requests:
                values = []
                for index, array in enumerate(arrays):
                    value = cp.asarray(array[request.field], dtype=cp.float32)
                    if request.field == "refl" and value.ndim == 3:
                        value = cp.max(value, axis=0)
                    if request.field == "rain_total" and rows[order[index]][0] == "history":
                        for key in ("RAINC", "RAINSH"):
                            if key in array:
                                value = value + cp.asarray(array[key])
                        origin, origin_time = rain_origin[(domain, order[index])]
                        value = value - origin
                    values.append(value)
                stack = cp.ascontiguousarray(cp.stack(values))
                for name, products, attrs in reduce_family_frame({request.field: stack}, (request,), roster,
                        available_bytes=int(cp.cuda.runtime.memGetInfo()[0]) - (256 << 20), array_module=cp):
                    stamp = valid.replace(":", "-")
                    path = out / domain / "family-products" / name / request.field / f"{stamp}.nc"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if request.field == "rain_total":
                        attrs["precipitation_baseline"] = json.dumps({str(i): rain_origin[(domain, i)][1]
                            for i in order if (domain, i) in rain_origin})
                    write_product_frame(path, products, valid_time=valid, latitude=lat, longitude=lon,
                                        global_attrs=dict(attrs, input_receipt=str(source.resolve())))
                    paths.setdefault(name, {})[request.field] = str(path)
                    if request.field == "refl":
                        for statistic in ("mean", "pmm"):
                            forecasts[(name, statistic)] = cp.asnumpy(products.outputs[f"refl:{statistic}"])
                        drawn, _ = rustwx.run_ensemble_product_renderer(renderer, path, out_dir=out / "maps" / name,
                            domain=domain, fields=("refl",), products=("prob", "mean", "pmm"),
                            width=1000, height=700, source_label=(f"{name} mass {roster['family_weights'][name]:g}"
                                if name != "combined" else "combined; masses in receipt"))
                        maps[name] = [str(p) for p in drawn]
                del stack, values, products
            frame = dict(domain=domain, valid_time=valid, products=paths, maps=maps,
                         unavailable_fields=missing, read_reduce_render_seconds=time.perf_counter() - started)
            # Archive errors remain explicit: never replace missing MRMS with zero.
            try:
                scan = cache.ensure(valid)
                obslat, obslon = read_geo_pack(cache.geometry_path)
                observed = read_grid_pack(scan.pack_path)
                plan = build_plan(source_latitude=obslat, source_longitude=obslon,
                    destination_latitude=lat, destination_longitude=lon, method="nearest", max_distance_m=4500.)
                obs, obsvalid = apply_plan(plan, observed.array("values"), observed.array("valid").astype(bool))
                scores = score_reflectivity(forecasts, obs, obsvalid, dx_m=dx_m, half_widths=half_widths)
                result["scores"].extend(dict(row, valid_time=valid, domain=domain) for row in scores)
                frame["observation"] = scan.record()
                frame["regrid"] = plan.record()
                frame["sheets"] = render_comparison(out, domain, valid, maps, obs, obsvalid,
                                                     lat, lon, renderer)
                frame["status"] = "scored"
            except (LookupError, RuntimeError) as error:
                from woof.obs.mrms_fetch import MrmsFrameAbsent, MrmsArchiveUnavailable
                if not isinstance(error, (MrmsFrameAbsent, MrmsArchiveUnavailable)):
                    raise
                frame.update(status="observation-unavailable", error=str(error))
            frame["total_seconds"] = time.perf_counter() - started
            result["frames"].append(frame)
            (out / "family-products.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["frames"]:
        raise ValueError("no selected output hour; use --valid-time for a saved sub-hour frame")
    if result["scores"]:
        with (out / "scores.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["scores"][0]))
            writer.writeheader()
            writer.writerows(result["scores"])
    return result


def render_comparison(out, domain, valid, maps, observed, valid_mask, latitude, longitude, renderer):
    """Render MRMS through the same Rust field painter, then compose panels."""
    from woof import rustwx
    values = np.where(valid_mask, observed, np.nan).astype(np.float32)
    # A one-plane observation uses the same typed product transport, retaining
    # its MRMS source label. This is an observation panel, not an ensemble mean.
    products = SimpleNamespace(calls=1, _xp=np, members=1, fields={"refl": values[None]},
        requests=(FieldProducts("refl", "dBZ"),), outputs={f"refl:{kind}": values
            for kind in ("mean", "spread", "min", "max", "finite_count")})
    products.outputs["refl:finite_count"] = valid_mask.astype(np.uint32)
    stamp = valid.replace(":", "-")
    path = out / domain / "family-products" / "mrms" / f"{stamp}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_product_frame(path, products, valid_time=valid, latitude=latitude, longitude=longitude,
                        global_attrs={"observation_source": "MRMS composite reflectivity"})
    panels, _ = rustwx.run_ensemble_product_renderer(renderer, path, out_dir=out / "maps" / "mrms",
        domain=domain, fields=("refl",), products=("mean",), width=1000, height=700,
        source_label="MRMS observation")
    rows = []
    names = [name for name in sorted(maps) if name != "combined"] + ["combined"]
    thresholds = sorted({Path(p).parent.parent.name for p in maps["combined"] if "ens_prob_" in Path(p).name})
    for slug in thresholds:
        rows.append([str(panels[0])] + [next(p for p in maps[name] if Path(p).parent.parent.name == slug) for name in names])
    # PMM beside MRMS makes displacement and reflectivity intensity visible.
    rows.append([str(panels[0])] + [next(p for p in maps[name] if "ens_pmm_" in Path(p).name) for name in names])
    sheet = out / domain / "family-comparison" / valid[:10] / f"{stamp}.png"
    rustwx.compose_ensemble_sheet(renderer, rows, sheet,
        title="MRMS | family probabilities | combined; final row: PMM (dBZ)", valid_time=valid)
    return [str(sheet)]


def add_arguments(parser):
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--member-map", type=Path)
    parser.add_argument("--dx-m", type=float, default=3000.)
    parser.add_argument("--mrms-cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--valid-time", action="append")
    return parser


def _run(args):
    result = run_products(args.run, args.out,
        member_map=json.loads(args.member_map.read_text()) if args.member_map else None,
        dx_m=args.dx_m, mrms_cache=args.mrms_cache, offline=args.offline,
        renderer=args.renderer, hours=set(args.valid_time) if args.valid_time else None)
    print(json.dumps({"receipt": str(args.out / "family-products.json"), "frames": len(result["frames"]),
                      "scores": len(result["scores"])}))
    return 0 if all(f["status"] == "scored" for f in result["frames"]) else 2


def register_cli(subparsers):
    parser = subparsers.add_parser("ensemble-products", help="GPU family products and MRMS scores from an existing ensemble run")
    add_arguments(parser)
    parser.set_defaults(func=_run)


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
