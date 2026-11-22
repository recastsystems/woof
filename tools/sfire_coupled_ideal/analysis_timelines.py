"""Analysis charts and tabular timelines from native Rust comparison metrics.

No gridded weather field is decoded, transformed or plotted here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    document = json.loads(path.read_text())
    if document["schema"] != "sfire-ideal-comparison-v1":
        raise ValueError("analysis requires the native Rust comparison schema")
    times = [frame["time_seconds"] for frame in document["frames"]]
    if times != list(range(0, 3601, 120)):
        raise ValueError("analysis requires all 31 official two-minute history frames")
    return document


def rows(model: str, frames):
    result = []
    sensible = latent = 0.0
    initial = frames[0]
    for index, frame in enumerate(frames):
        if index:
            previous = frames[index - 1]
            dt = frame["time_seconds"] - previous["time_seconds"]
            sensible += 0.5 * dt * (frame["sensible_power_w"] + previous["sensible_power_w"])
            latent += 0.5 * dt * (frame["latent_power_w"] + previous["latent_power_w"])
        row = {"model": model, "time_seconds": frame["time_seconds"],
               "fire_area_m2": frame["fire_area_m2"], "perimeter_m": frame["perimeter_m"],
               "sensible_power_w": frame["sensible_power_w"], "latent_power_w": frame["latent_power_w"],
               "cumulative_sensible_j_trapezoid": sensible, "cumulative_latent_j_trapezoid": latent,
               "updraft_max_m_s": frame["fields"]["W"]["max"],
               "vertical_speed_abs_max_m_s": max(abs(frame["fields"]["W"]["min"]), abs(frame["fields"]["W"]["max"])),
               "vertical_velocity_rms_m_s": frame["fields"]["W"]["rms"],
               "mass_weighted_theta_k": frame["mass_weighted_theta_k"],
               "theta_change_k": frame["mass_weighted_theta_k"] - initial["mass_weighted_theta_k"],
               "dry_air_mass_kg": frame["dry_air_mass_kg"],
               "dry_air_mass_change_kg": frame["dry_air_mass_kg"] - initial["dry_air_mass_kg"],
               "vapor_mass_kg": frame["water_vapor_mass_kg"],
               "vapor_mass_change_kg": frame["water_vapor_mass_kg"] - initial["water_vapor_mass_kg"]}
        if any(not math.isfinite(value) for key, value in row.items() if key != "model"):
            raise ValueError("analysis refused a nonfinite scalar")
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corrected", type=Path)
    parser.add_argument("--strict", type=Path)
    parser.add_argument("--default-receipt", type=Path)
    parser.add_argument("--source-receipt", type=Path,
                        help="External immutable archive/diff authority beside source-files.sha256.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = load(args.original)
    series = {"WRF original": rows("wrf_original", [f["reference"] for f in original["frames"]]),
              "WOOF-ArWen default": rows("arwen_default", [f["candidate"] for f in original["frames"]])}
    sources = {args.original.name: sha(args.original)}
    default_tip = None
    if args.default_receipt:
        receipt = json.loads(args.default_receipt.read_text())
        if not receipt["completed"] or receipt["step_count"] != 7200 or receipt["final_time_seconds"] != 3600:
            raise ValueError("analysis default receipt does not prove the complete official window")
        provenance = receipt.get("source_provenance")
        if provenance is not None:
            if provenance["dirty_selected_paths"]:
                raise ValueError("analysis default source must be a committed snapshot")
            default_tip = provenance["git_tip"]
        elif args.source_receipt:
            authority = json.loads(args.source_receipt.read_text())
            manifest_path = args.source_receipt.parent / "source-files.sha256.json"
            if authority["status"] != "PASS" or sha(manifest_path) != authority["source_manifest_sha256"]:
                raise ValueError("external source authority does not bind its complete manifest")
            manifest = json.loads(manifest_path.read_text())
            expected = {name: digest for name, digest in manifest.items()
                        if name.startswith("woof/") and Path(name).suffix in (".py", ".cu", ".cuh")}
            if expected != receipt["sources_sha256"]:
                raise ValueError("executed engine code differs from the external committed source manifest")
            default_tip = authority["tip"]
            sources[args.source_receipt.name] = sha(args.source_receipt)
            sources[manifest_path.name] = sha(manifest_path)
        else:
            raise ValueError("analysis requires an immutable source authority for the complete default run")
        sources[args.default_receipt.name] = sha(args.default_receipt)
    if args.corrected:
        corrected = load(args.corrected)
        series["WRF corrected"] = rows("wrf_corrected", [f["reference"] for f in corrected["frames"]])
        sources[args.corrected.name] = sha(args.corrected)
    if args.strict:
        strict = load(args.strict)
        series["WOOF-ArWen strict control"] = rows("arwen_strict", [f["candidate"] for f in strict["frames"]])
        sources[args.strict.name] = sha(args.strict)
    for side, label in (("reference", "WRF original"), ("candidate", "WOOF-ArWen default")):
        for prefix in ("sensible", "latent"):
            estimated = series[label][-1][f"cumulative_{prefix}_j_trapezoid"]
            native = original[f"{side}_integral"][f"{prefix}_heat_j_trapezoid"]
            if not math.isclose(estimated, native, rel_tol=1e-14):
                raise ValueError("analysis integration disagrees with the native Rust receipt")
    csv_path = args.output / "coupled-ideal-timelines.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(next(iter(series.values()))[0]))
        writer.writeheader()
        for values in series.values():
            writer.writerows(values)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.22})
    colors = {"WRF original": "#2563eb", "WRF corrected": "#059669", "WOOF-ArWen default": "#ea580c", "WOOF-ArWen strict control": "#7c3aed"}
    styles = {"WRF original": "-", "WRF corrected": "--", "WOOF-ArWen default": "-", "WOOF-ArWen strict control": ":"}

    def figure(filename, panels, height):
        fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(13.2, height), squeeze=False)
        for ax, (field, scale, title) in zip(axes.flat, panels):
            for label, values in series.items():
                ax.plot([row["time_seconds"] / 60 for row in values],
                        [row[field] * scale for row in values], label=label,
                        color=colors[label], linestyle=styles[label], linewidth=2.0)
            ax.set_title(title, loc="left")
            ax.set_xlabel("Elapsed time (min)")
            ax.set_xlim(0, 60)
        fig.suptitle("Coupled SFIRE ideal case", fontsize=17, x=0.07, ha="left", y=1 - 0.12 / height)
        fig.text(0.07, 1 - 0.47 / height, "WRF 4.7.1 em_fire | 50 m atmosphere | 12.5 m fire | 120 s histories",
                 fontsize=11, va="top")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if default_tip:
            labels = ["WOOF-ArWen strict, earlier source" if label == "WOOF-ArWen strict control" else label
                      for label in labels]
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1 - 0.83 / height), frameon=False)
        fig.tight_layout(rect=(0.025, 0.015, 0.985, 1 - 1.5 / height), h_pad=2.0, w_pad=2.5)
        fig.savefig(args.output / filename, dpi=180, facecolor="white")
        plt.close(fig)

    figure("fire-timelines.png", (("fire_area_m2", 1e-6, "Fire area (km²)"),
            ("perimeter_m", 1e-3, "Fire perimeter (km)"),
            ("sensible_power_w", 1e-6, "Sensible heat power (MW)"),
            ("latent_power_w", 1e-6, "Latent heat power (MW)"),
            ("cumulative_sensible_j_trapezoid", 1e-12, "Cumulative sensible heat (TJ)"),
            ("cumulative_latent_j_trapezoid", 1e-12, "Cumulative latent heat (TJ)")), 11.5)
    figure("atmosphere-timelines.png", (("updraft_max_m_s", 1.0, "Maximum upward velocity (m/s)"),
            ("vertical_velocity_rms_m_s", 1.0, "RMS vertical velocity (m/s)"),
            ("theta_change_k", 1.0, "Mean potential temperature change (K)"),
            ("vapor_mass_change_kg", 1e-3, "Water vapor mass change (tonnes)")), 8.7)
    caption = ("Time series from the 31 complete history frames of the compiled WRF ideal case and the ordinary WOOF-ArWen run. "
               "The original WRF case retains its source unchanged. WOOF-ArWen uses the corrected atmosphere-aligned fire clock and follows native WRF at the open-boundary top row. "
               "The corrected WRF control applies that source fix in a separate copied reference. "
               "WOOF-ArWen transports dry potential temperature, while native WRF uses its moist-theta formulation. "
               "Cumulative heat is a trapezoidal estimate at the stored 120-second cadence. "
               "The mass series includes open-boundary exchange. These are implementation comparisons; observed fire perimeters determine forecast skill.\n")
    if not args.corrected:
        caption += "The corrected native reference is still running and is not drawn in this version.\n"
    if args.strict:
        caption += "The strict WOOF-ArWen line is a separate arithmetic diagnostic and does not qualify the default path.\n"
        if default_tip:
            caption += "The strict line retains the earlier source snapshot recorded in its own run receipt. It is not a strict execution of the later default source tip.\n"
    if default_tip:
        caption += f"The qualified default WOOF-ArWen source tip is {default_tip}.\n"
    if args.default_receipt and receipt["case_semantics"].get("native_tracer_opt") == 3:
        caption += ("The separately hashed qualification harness imports and transports native bulk smoke option 3. "
                    "Original WRF's first-column-only smoke insertion defect is reported separately; "
                    "the plots compare fire and atmospheric fields.\n")
    (args.output / "CAPTIONS.txt").write_text(caption)
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in (csv_path, args.output / "fire-timelines.png", args.output / "atmosphere-timelines.png", args.output / "CAPTIONS.txt")}
    qa = {"schema": "sfire-ideal-analysis-qa-v1", "source_sha256": sources,
          "default_git_tip": default_tip,
          "models": list(series), "samples_per_model": 31, "start_seconds": 0, "end_seconds": 3600,
          "finite_numeric_values": True, "native_integral_agreement": True,
          "weather_field_maps": False, "matplotlib_version": matplotlib.__version__, "artifacts": artifacts}
    (args.output / "analysis-qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps({"models": list(series), "samples_per_model": 31, "artifacts": artifacts}, sort_keys=True))


if __name__ == "__main__":
    main()
