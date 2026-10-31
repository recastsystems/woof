"""Write an evidence-linked phase-two section after every full control ends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(root, name):
    return json.loads((root / name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--original-comparison", default="original-wrf-vs-arwen-default-full.json")
    parser.add_argument("--corrected-comparison", default="corrected-wrf-vs-arwen-default-full.json")
    parser.add_argument("--default-run", default="arwen-default-full-receipt.json")
    args = parser.parse_args()
    root = args.evidence
    original = read(root, args.original_comparison)
    corrected = read(root, args.corrected_comparison)
    strict = read(root, "original-wrf-vs-arwen-strict-full.json")
    default_run = read(root, args.default_run)
    strict_run = read(root, "arwen-strict-full-receipt.json")
    for name in ("reference-o2-final-receipt.json", "reference-corrected-final-receipt.json"):
        if not read(root, name)["completed_successfully"]:
            raise ValueError("the native reference receipt does not prove successful completion")
    for run in (default_run, strict_run):
        if not run["completed"] or run["step_count"] != 7200 or run["final_time_seconds"] != 3600:
            raise ValueError("the WOOF receipt does not prove the complete native window")
    arms = (("WRF original", original["frames"][-1]["reference"], original["reference_integral"]),
            ("WRF corrected", corrected["frames"][-1]["reference"], corrected["reference_integral"]),
            ("WOOF default", original["frames"][-1]["candidate"], original["candidate_integral"]),
            ("WOOF strict control, earlier source", strict["frames"][-1]["candidate"], strict["candidate_integral"]))
    text = ["The complete coupled ideal case runs end to end under ordinary WOOF arithmetic.\n",
            f"Compiled WRF original and corrected references each completed 3600 seconds. WOOF default completed 7200 steps in {default_run['wall_seconds']:.3f} seconds with 31 Rust-written histories. The separate strict control also completed 7200 steps. Evidence: `evidence/ideal/reference-o2-final-receipt.json`, `reference-corrected-final-receipt.json`, `{args.default_run}`, `arwen-strict-full-receipt.json`.\n",
            "| Arm | Final fire area (m2) | Perimeter (m) | Sensible heat (TJ) | Latent heat (TJ) | Maximum upward velocity (m/s) |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, frame, integral in arms:
        text.append(f"| {label} | {frame['fire_area_m2']:.3f} | {frame['perimeter_m']:.3f} | {integral['sensible_heat_j_trapezoid']/1e12:.6f} | {integral['latent_heat_j_trapezoid']/1e12:.6f} | {frame['fields']['W']['max']:.6f} |")
    text.append("\nHeat totals use trapezoidal integration at the stored 120-second history cadence. They are estimates, not every-step accumulation. The physical fire domain is 408 by 408 cells; native terminal extension cells are excluded. Perimeters use the Rust contour engine.\n")
    text.append("\nThe native full reference uses scalar `-O2 -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize` CPU arithmetic. Its six stored frames through 600 seconds compare 28,089,576 FP32 words against the preserved scalar O0 run: all words match through 480 seconds, then two level-set words differ by 1 ULP at 600 seconds. Disassembly identifies the compiler's replacement of native `REAL ** 2.0` with multiplication in reinitialization. This optimized full run is not claimed bit-identical to the scalar routine oracle. Evidence: `evidence/ideal/o0-o2-postignition-control.json`, `reinitialization-arithmetic-control.txt`, and the compiler and source fields in the original final receipt.\n")
    for label, document, filename in (("original WRF", original, args.original_comparison),
                                      ("corrected WRF", corrected, args.corrected_comparison)):
        final = document["frames"][-1]
        pct = 100 * final["area_difference_m2"] / final["reference"]["fire_area_m2"]
        text.append(f"Against {label}, default WOOF final area differs by {pct:+.4f}%, mean perimeter distance is {final['perimeter_midpoint_mean_distance_m']:.6f} m, sampled Hausdorff distance is {final['perimeter_sampled_hausdorff_m']:.6f} m, and burned-cell intersection over union is {final['burned_cell_iou']:.6f}. Evidence: `evidence/ideal/{filename}`.\n")
    final = corrected["frames"][-1]["differences"]
    text.extend(["\n| Default WOOF versus corrected WRF | Final RMSE |", "|---|---:|"])
    for name, unit in (("T", "K"), ("U", "m/s"), ("V", "m/s"), ("W", "m/s"), ("QVAPOR", "kg/kg"), ("P", "Pa")):
        text.append(f"| {name} | {final[name]['rmse']:.9g} {unit} |")
    dp = original["frames"][0]["differences"]["P"]["rmse"]
    sp = strict["frames"][0]["differences"]["P"]["rmse"]
    text.append(f"\nAll compared initial prognostic and fire words match native WRF's first history frame. Default diagnostic pressure differs with initial RMSE {dp:.9g} Pa. The retained earlier-source arithmetic diagnostic has initial RMSE {sp:.9g} Pa; its paired ordinary run is preserved separately. The full coupled atmosphere is not word-identical. WOOF evolves dry potential temperature; native WRF retains `use_theta_m=1`. These numerical formulations and the declared source corrections must remain distinct from routine-level parity. Evidence: the initial/final field grades in the original/default and original/strict comparison JSON files.\n")
    text.extend(["\n| Arm | Dry air mass change (kg) | Water vapor mass change (kg) | Mean potential temperature change (K) |",
                 "|---|---:|---:|---:|"])
    for label, _, integral in arms:
        text.append(f"| {label} | {integral['dry_air_mass_change_kg']:.3f} | {integral['vapor_mass_change_kg']:.3f} | {integral['theta_change_k']:.9g} |")
    vapor_delta = corrected["candidate_integral"]["vapor_mass_change_kg"] - corrected["reference_integral"]["vapor_mass_change_kg"]
    initial_vapor = corrected["frames"][0]["reference"]["water_vapor_mass_kg"]
    text.append(f"\nAll stored fields and aggregate values are finite. The domain has open atmospheric boundaries, so these mass changes are not closed-domain conservation errors. Default WOOF's net vapor change differs from corrected WRF by {vapor_delta:.3f} kg, or {100 * vapor_delta / initial_vapor:.6f}% of the initial vapor mass. Net vapor change diverges after 30 minutes. The earlier-source strict control also retains a net vapor difference; it is not a same-source full-model comparison with the newly qualified default run. This remains an atmospheric qualification limit, and the comparison does not separate every cause. Evidence: the integral and profile sections of the original/default, corrected/default and original/strict comparison JSON files.\n")
    text.append("\nThe copied native reference corrects the one-step-ahead fire clock. Source and executable hashes, exact compiler/link commands and unchanged-library reuse are recorded in `evidence/ideal/SFIRE_CORRECTIONS.json`, `SFIRE_CORRECTIONS.patch`, `incremental-build-receipt.json`, and the corrected final receipt. Original source, inputs and binaries remain preserved.\n")
    text.append("\nAnalysis-only charts, CSV, captions and numeric/hash QA are kept with the run's receipts. No weather-field map was drawn with matplotlib. Observed fire perimeters remain the skill reference; this ideal comparison does not establish real-fire forecast skill.\n")
    provenance = default_run.get("source_provenance")
    if provenance:
        if provenance["dirty_selected_paths"]:
            raise ValueError("final qualification cannot name a dirty snapshot as a committed tip")
        text.append(f"\nThe qualified default execution uses committed source `{provenance['git_tip']}`. Its staged engine code manifest was verified before execution, and the run receipt records the source, harness and output hashes. The strict run retains its older source manifest and is an arithmetic diagnostic, not a strict execution of this later tip. The preceding default and strict controls remain preserved.\n")
    else:
        text.append("\nExecution source manifests are saved per run. Current checkout changes are listed in `evidence/ideal/current-vs-executed-source.json`; these measured runs must not be described as executions of a later source tip until its final short/full qualification runs finish.\n")
    args.output.write_text("\n".join(text))


if __name__ == "__main__":
    main()
