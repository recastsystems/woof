#!/usr/bin/env python3
"""Compare WOOF-ArWen and stock-WRF two-way nest feedback signatures.

This is deliberately a signature comparison, not a bitwise or amplitude
certification.  Each feedback delta is ``feedback=1 - feedback=0`` on the
parent-domain cells covered by the child interior.  The report records
max/mean absolute deltas and checks whether the two spatial delta vectors
point in the same direction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Iterable

import netCDF4
import numpy as np


DEFAULT_FIELDS = (
    "MU", "U", "V", "W", "T", "PH",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    "QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
    "QNDROP", "QNCCN", "QVGRAUPEL", "QVHAIL",
)

# WRF Registry.EM_COMMON fields carried by the non-averaging feedback
# routines.  These are limited to fields ArWen can write and that are useful
# to a feedback comparison; DEFAULT_FIELDS deliberately contains none of
# them because ArWen's live feedback inventory is dynamics/moisture only.
WRF_COPY_FCNM_FIELDS = frozenset({
    "LU_INDEX", "TSLB", "SMOIS", "SH2O", "XICE", "VEGFRA",
    "ACSNOW", "ACRUNOFF", "ACSNOM", "SNOW", "SNOWH", "CANWAT",
    "TSK", "TMN", "XLAND", "SNOWC",
})
WRF_COPY_FCNI_FIELDS = frozenset({
    "IVGTYP", "ISLTYP", "I_RAINC", "I_RAINNC",
})


@dataclass(frozen=True)
class ChildGeometry:
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    child_nx: int
    child_ny: int


@dataclass(frozen=True)
class Frame:
    path: Path
    time_index: int


def _wrfout_paths(run_dir: Path, domain_id: int) -> list[Path]:
    paths = sorted(
        path for path in run_dir.rglob(f"wrfout_d{domain_id:02d}*")
        if path.is_file())
    if not paths:
        raise ValueError(
            f"{run_dir} has no wrfout_d{domain_id:02d} parent/child files")
    return paths


def _text_times(dataset: netCDF4.Dataset) -> tuple[str, ...]:
    if "Times" not in dataset.variables:
        raise ValueError(f"{dataset.filepath()} has no Times variable")
    raw = netCDF4.chartostring(np.asarray(dataset.variables["Times"][:]))
    return tuple(
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in np.atleast_1d(raw))


def _frame_index(run_dir: Path) -> dict[str, Frame]:
    frames: dict[str, Frame] = {}
    for path in _wrfout_paths(run_dir, 1):
        with netCDF4.Dataset(path) as dataset:
            for time_index, valid_time in enumerate(_text_times(dataset)):
                if valid_time in frames:
                    raise ValueError(
                        f"{run_dir} has duplicate parent time {valid_time}")
                frames[valid_time] = Frame(path, time_index)
    return frames


def _child_geometry(run_dir: Path) -> ChildGeometry:
    path = _wrfout_paths(run_dir, 2)[0]
    with netCDF4.Dataset(path) as dataset:
        required = (
            "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO")
        missing = [name for name in required
                   if name not in dataset.ncattrs()]
        if missing:
            raise ValueError(
                f"{path} lacks child topology attributes {missing}")
        return ChildGeometry(
            i_parent_start=int(dataset.I_PARENT_START),
            j_parent_start=int(dataset.J_PARENT_START),
            parent_grid_ratio=int(dataset.PARENT_GRID_RATIO),
            child_nx=len(dataset.dimensions["west_east"]),
            child_ny=len(dataset.dimensions["south_north"]))


def _horizontal_stagger(variable) -> tuple[bool, bool]:
    dims = set(variable.dimensions)
    x_staggered = "west_east_stag" in dims
    y_staggered = "south_north_stag" in dims
    if x_staggered and y_staggered:
        raise ValueError(
            f"{variable.name} is staggered in both horizontal directions")
    return x_staggered, y_staggered


def _operator_descriptor(
        field: str, *, ratio: int,
        x_staggered: bool, y_staggered: bool,
        ) -> dict[str, object]:
    """Classify one field using WRF v4.6.1's feedback operator split."""
    if ratio <= 0:
        raise ValueError(f"invalid parent_grid_ratio {ratio}")
    if field in WRF_COPY_FCNM_FIELDS:
        return {
            "class": "masked",
            "routine": "copy_fcnm",
            "stagger": (
                "x" if x_staggered else "y" if y_staggered else "mass"),
            "stencil": (
                "coincident-point" if ratio % 2
                else "nearest-neighbor-southwest"),
            "source_points_per_parent": 1,
        }
    if field in WRF_COPY_FCNI_FIELDS:
        return {
            "class": "masked",
            "routine": "copy_fcni",
            "stagger": (
                "x" if x_staggered else "y" if y_staggered else "mass"),
            "stencil": (
                "coincident-point" if ratio % 2
                else "nearest-neighbor-southwest"),
            "source_points_per_parent": 1,
        }
    if x_staggered:
        return {
            "class": "u-face",
            "routine": "copy_fcn",
            "stagger": "x",
            "stencil": "face-average",
            "source_points_per_parent": ratio,
        }
    if y_staggered:
        return {
            "class": "v-face",
            "routine": "copy_fcn",
            "stagger": "y",
            "stencil": "face-average",
            "source_points_per_parent": ratio,
        }
    return {
        "class": "mass",
        "routine": "copy_fcn",
        "stagger": "mass",
        "stencil": "cell-average",
        "source_points_per_parent": ratio * ratio,
    }


def _overlap_slices(
        geometry: ChildGeometry, *, spec_zone: int,
        x_staggered: bool, y_staggered: bool,
        ) -> tuple[slice, slice]:
    ratio = geometry.parent_grid_ratio
    if ratio <= 0:
        raise ValueError(f"invalid parent_grid_ratio {ratio}")
    if geometry.child_nx % ratio or geometry.child_ny % ratio:
        raise ValueError(
            "child mass dimensions must be divisible by parent_grid_ratio: "
            f"{geometry.child_nx}x{geometry.child_ny} / {ratio}")
    # WRF copy_fcn bounds are one-based and inclusive:
    #   ips + spec_zone .. ips + child_n/ratio - 1 - spec_zone.
    # A coincident high-side face is included for the matching stagger.
    i_lo = geometry.i_parent_start + spec_zone
    j_lo = geometry.j_parent_start + spec_zone
    i_hi = (geometry.i_parent_start + geometry.child_nx // ratio
            - 1 - spec_zone + int(x_staggered))
    j_hi = (geometry.j_parent_start + geometry.child_ny // ratio
            - 1 - spec_zone + int(y_staggered))
    if i_hi < i_lo or j_hi < j_lo:
        raise ValueError(
            "child interior is empty after excluding its boundary zone")
    return slice(j_lo - 1, j_hi), slice(i_lo - 1, i_hi)


def _read_overlap(
        frame: Frame, field: str, geometry: ChildGeometry, spec_zone: int,
        ) -> tuple[np.ndarray, dict[str, object]]:
    with netCDF4.Dataset(frame.path) as dataset:
        if field not in dataset.variables:
            raise KeyError(field)
        variable = dataset.variables[field]
        x_staggered, y_staggered = _horizontal_stagger(variable)
        operator = _operator_descriptor(
            field, ratio=geometry.parent_grid_ratio,
            x_staggered=x_staggered, y_staggered=y_staggered)
        y_slice, x_slice = _overlap_slices(
            geometry, spec_zone=spec_zone,
            x_staggered=x_staggered, y_staggered=y_staggered)
        values = np.ma.asarray(variable[frame.time_index])
        values = np.asarray(values.filled(np.nan), dtype=np.float64)
        if values.ndim < 2:
            raise ValueError(
                f"{frame.path}:{field} has no horizontal plane")
        return values[..., y_slice, x_slice], operator


def _delta(
        frames0: dict[str, Frame], frames1: dict[str, Frame],
        valid_time: str, field: str, geometry: ChildGeometry, spec_zone: int,
        ) -> tuple[np.ndarray, dict[str, object]]:
    before, operator0 = _read_overlap(
        frames0[valid_time], field, geometry, spec_zone)
    after, operator1 = _read_overlap(
        frames1[valid_time], field, geometry, spec_zone)
    if operator0 != operator1:
        raise ValueError(
            f"{field} feedback pair operator classes differ: "
            f"{operator0} vs {operator1}")
    if before.shape != after.shape:
        raise ValueError(
            f"{field} feedback pair shape mismatch: "
            f"{before.shape} vs {after.shape}")
    return after - before, operator0


def _signature_row(
        arwen_delta: np.ndarray, wrf_delta: np.ndarray, *,
        field: str, valid_time: str, elapsed_hours: float,
        operator: dict[str, object],
        cosine_threshold: float, inactive_tolerance: float,
        ) -> dict[str, object]:
    if arwen_delta.shape != wrf_delta.shape:
        raise ValueError(
            f"{field} WOOF-ArWen/WRF overlap shapes differ: "
            f"{arwen_delta.shape} vs {wrf_delta.shape}")
    finite = np.isfinite(arwen_delta) & np.isfinite(wrf_delta)
    if not finite.any():
        raise ValueError(f"{field} has no finite common overlap values")
    arwen = arwen_delta[finite].ravel()
    wrf = wrf_delta[finite].ravel()
    arwen_norm = float(np.linalg.norm(arwen))
    wrf_norm = float(np.linalg.norm(wrf))
    arwen_max = float(np.max(np.abs(arwen)))
    wrf_max = float(np.max(np.abs(wrf)))
    wrf_active = wrf_max > inactive_tolerance
    arwen_active = arwen_max > inactive_tolerance
    if wrf_norm > 0.0 and arwen_norm > 0.0:
        cosine = float(np.dot(arwen, wrf) / (arwen_norm * wrf_norm))
        gain = float(np.dot(arwen, wrf) / np.dot(wrf, wrf))
        residual = arwen - gain * wrf
        normalized_residual = float(
            np.linalg.norm(residual) / arwen_norm)
    else:
        cosine = None
        gain = None
        normalized_residual = None
    tracks = (
        (not wrf_active and not arwen_active)
        or (wrf_active and arwen_active
            and cosine is not None and cosine >= cosine_threshold)
    )
    return {
        "field": field,
        "operator": operator,
        "valid_time": valid_time,
        "elapsed_hours": elapsed_hours,
        "points": int(arwen.size),
        "arwen": {
            "max_abs_delta": arwen_max,
            "mean_abs_delta": float(np.mean(np.abs(arwen))),
            "mean_signed_delta": float(np.mean(arwen)),
        },
        "wrf": {
            "max_abs_delta": wrf_max,
            "mean_abs_delta": float(np.mean(np.abs(wrf))),
            "mean_signed_delta": float(np.mean(wrf)),
        },
        "signature": {
            "cosine_similarity": cosine,
            "arwen_per_wrf_gain": gain,
            "normalized_residual_after_gain": normalized_residual,
            "tracks": bool(tracks),
        },
    }


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d_%H:%M:%S")


def compare_runs(
        arwen_feedback0: Path, arwen_feedback1: Path,
        wrf_feedback0: Path, wrf_feedback1: Path, *,
        fields: Iterable[str] = DEFAULT_FIELDS, spec_zone: int = 1,
        cosine_threshold: float = 0.5, inactive_tolerance: float = 0.0,
        ) -> dict[str, object]:
    """Return the machine-readable four-run feedback signature report."""
    run_dirs = {
        "arwen_feedback0": Path(arwen_feedback0),
        "arwen_feedback1": Path(arwen_feedback1),
        "wrf_feedback0": Path(wrf_feedback0),
        "wrf_feedback1": Path(wrf_feedback1),
    }
    indexes = {name: _frame_index(path)
               for name, path in run_dirs.items()}
    common_times = sorted(set.intersection(
        *(set(index) for index in indexes.values())))
    if not common_times:
        raise ValueError("the four parent runs have no common valid time")
    geometries = {
        name: _child_geometry(path) for name, path in run_dirs.items()}
    for pair in (("arwen_feedback0", "arwen_feedback1"),
                 ("wrf_feedback0", "wrf_feedback1")):
        if geometries[pair[0]] != geometries[pair[1]]:
            raise ValueError(
                f"{pair[0]}/{pair[1]} child geometries differ")

    origin = _parse_time(common_times[0])
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for valid_time in common_times:
        elapsed_hours = (
            _parse_time(valid_time) - origin).total_seconds() / 3600.0
        for field in fields:
            try:
                arwen_delta, arwen_operator = _delta(
                    indexes["arwen_feedback0"],
                    indexes["arwen_feedback1"], valid_time, field,
                    geometries["arwen_feedback0"], spec_zone)
                wrf_delta, wrf_operator = _delta(
                    indexes["wrf_feedback0"],
                    indexes["wrf_feedback1"], valid_time, field,
                    geometries["wrf_feedback0"], spec_zone)
            except KeyError:
                skipped.append({
                    "field": field, "valid_time": valid_time,
                    "reason": "field is not present in every run",
                })
                continue
            if arwen_operator != wrf_operator:
                raise ValueError(
                    f"{field} WOOF-ArWen/WRF operator classes differ: "
                    f"{arwen_operator} vs {wrf_operator}")
            rows.append(_signature_row(
                arwen_delta, wrf_delta, field=field,
                valid_time=valid_time, elapsed_hours=elapsed_hours,
                operator=arwen_operator,
                cosine_threshold=cosine_threshold,
                inactive_tolerance=inactive_tolerance))
    if not rows:
        raise ValueError("none of the requested fields exists in all four runs")
    return {
        "schema": "gpuwm-feedback-signature-comparison-v2",
        "comparison": "parent overlap feedback=1 minus feedback=0",
        "certification_kind": "signature-not-bit-comparison",
        "field_operator_classes": {
            row["field"]: row["operator"] for row in rows
        },
        "spec_zone": spec_zone,
        "cosine_threshold": cosine_threshold,
        "inactive_tolerance": inactive_tolerance,
        "run_directories": {
            name: str(path.resolve()) for name, path in run_dirs.items()},
        "geometry": {
            name: vars(geometry) for name, geometry in geometries.items()},
        "rows": rows,
        "skipped": skipped,
        "all_signatures_track": all(
            row["signature"]["tracks"] for row in rows),
    }


def _write_synthetic_run(
        run_dir: Path, *, delta: np.ndarray | None) -> None:
    run_dir.mkdir(parents=True)
    valid_time = "2000-01-01_00:00:00"
    parent_path = run_dir / "wrfout_d01_2000-01-01_00_00_00"
    with netCDF4.Dataset(parent_path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("bottom_top", 1)
        dataset.createDimension("south_north", 8)
        dataset.createDimension("west_east", 8)
        times = dataset.createVariable(
            "Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.frombuffer(
            valid_time.encode("ascii"), dtype="S1")
        field = dataset.createVariable(
            "T", "f4",
            ("Time", "bottom_top", "south_north", "west_east"))
        field[:] = 0.0
        if delta is not None:
            field[0, 0, 2:5, 2:5] = delta
    child_path = run_dir / "wrfout_d02_2000-01-01_00_00_00"
    with netCDF4.Dataset(child_path, "w") as dataset:
        dataset.I_PARENT_START = 2
        dataset.J_PARENT_START = 2
        dataset.PARENT_GRID_RATIO = 3
        dataset.createDimension("west_east", 15)
        dataset.createDimension("south_north", 15)


def synthetic_self_test() -> dict[str, object]:
    """Exercise geometry, differencing, statistics, and signature direction."""
    pattern = np.asarray(
        [[-2.0, -1.0, 0.0],
         [-1.0, 0.0, 1.0],
         [0.0, 1.0, 2.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-feedback-signature-") as temporary:
        root = Path(temporary)
        _write_synthetic_run(root / "arwen0", delta=None)
        _write_synthetic_run(root / "arwen1", delta=pattern * 2.0)
        _write_synthetic_run(root / "wrf0", delta=None)
        _write_synthetic_run(root / "wrf1", delta=pattern)
        report = compare_runs(
            root / "arwen0", root / "arwen1",
            root / "wrf0", root / "wrf1", fields=("T",),
            cosine_threshold=0.99)
    row = report["rows"][0]
    if not report["all_signatures_track"]:
        raise AssertionError("aligned synthetic feedback signature did not pass")
    if not np.isclose(row["signature"]["cosine_similarity"], 1.0):
        raise AssertionError("synthetic signature cosine is not one")
    if not np.isclose(row["signature"]["arwen_per_wrf_gain"], 2.0):
        raise AssertionError("synthetic signature gain is not two")
    opposite = _signature_row(
        pattern, -pattern, field="T", valid_time="synthetic",
        elapsed_hours=0.0,
        operator=_operator_descriptor(
            "T", ratio=3, x_staggered=False, y_staggered=False),
        cosine_threshold=0.5, inactive_tolerance=0.0)
    if opposite["signature"]["tracks"]:
        raise AssertionError("opposite synthetic feedback signature passed")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arwen-feedback0", type=Path)
    parser.add_argument("--arwen-feedback1", type=Path)
    parser.add_argument("--wrf-feedback0", type=Path)
    parser.add_argument("--wrf-feedback1", type=Path)
    parser.add_argument(
        "--fields", nargs="+", default=list(DEFAULT_FIELDS),
        help="WRF variable names to compare (default: feedback inventory)")
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--cosine-threshold", type=float, default=0.5)
    parser.add_argument("--inactive-tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.self_test:
        report = synthetic_self_test()
        print(json.dumps({
            "self_test": "PASS",
            "all_signatures_track": report["all_signatures_track"],
        }, indent=2, sort_keys=True))
        return 0
    required = {
        "--arwen-feedback0": args.arwen_feedback0,
        "--arwen-feedback1": args.arwen_feedback1,
        "--wrf-feedback0": args.wrf_feedback0,
        "--wrf-feedback1": args.wrf_feedback1,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    report = compare_runs(
        args.arwen_feedback0, args.arwen_feedback1,
        args.wrf_feedback0, args.wrf_feedback1,
        fields=args.fields, spec_zone=args.spec_zone,
        cosine_threshold=args.cosine_threshold,
        inactive_tolerance=args.inactive_tolerance)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["all_signatures_track"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
