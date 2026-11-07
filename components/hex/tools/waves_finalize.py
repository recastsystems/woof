"""Finalize a completed coupled wave case on either physics route using shared Rust tools.

The configuration names prepared, forecast, mesh, output, hours and label.
Missing final times or a failed atmosphere receipt refuse scoring.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    prepared, forecast, output = (Path(cfg[k]).resolve() for k in ("prepared", "forecast", "output"))
    manifest = json.loads((prepared / "manifest.json").read_text())
    native_receipt = forecast / "native-physics-receipt.json"
    if native_receipt.exists():
        atmosphere = json.loads(native_receipt.read_text())
        if atmosphere.get("completed") is not True or atmosphere["steps_completed"] != atmosphere["steps_requested"]:
            raise ValueError("native atmosphere did not complete every requested step")
        atmosphere_steps = atmosphere["steps_completed"]
        route = "mpas"
    else:
        atmosphere = json.loads((forecast / "cuda-v841-forecast-receipt.json").read_text())
        execution = atmosphere["forecast"]
        if (atmosphere.get("status") != "passed" or execution.get("refusal") or
                execution["steps_executed"] != execution["steps_requested"]):
            raise ValueError("WOOF atmosphere did not complete every requested step")
        atmosphere_steps = execution["steps_executed"]
        route = "arwen"
    if atmosphere_steps <= 0:
        raise ValueError("completed atmosphere must have positive step count")
    start = int(manifest["initial_valid_time"])
    seconds = float(cfg["hours"]) * 3600
    cadence = int(cfg.get("map_cadence_seconds", 21600))
    if seconds <= 0 or cadence <= 0 or seconds % cadence:
        raise ValueError("positive output cadence must divide the full forecast")
    end = start + int(seconds)
    expected = {datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                for t in range(start, end + 1, cadence)}
    frames = sorted((forecast / "waves" / "native").glob("*.nc"))
    actual = {p.stem for p in frames}
    if actual != expected:
        raise ValueError(f"wave fields do not cover the exact requested window: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    wave = json.loads((forecast / "waves" / "receipt.json").read_text())
    if datetime.fromisoformat(wave["valid_time"]).timestamp() != end:
        raise ValueError("wave receipt has not reached the final valid time")
    if int(wave["core"]["time_seconds"]) != end:
        raise ValueError("device wave clock differs from the requested final time")
    output.mkdir(parents=True, exist_ok=True)
    request = dict(operation="verify", prepared=str(prepared), output=str(output / "verification"),
                   threads=int(cfg.get("threads", 24)), models=[
        dict(label=cfg["label"], points=str(forecast / "waves" / "points.nc")),
        dict(label="GFS-Wave", points=str(prepared / "reference" / "points.nc"))])
    if cfg.get("reference_case"):
        request["models"][1] = dict(label="GFS-Wave replay", kind="station_ascii",
            case=cfg["reference_case"], fallback_points=str(prepared / "reference" / "points.nc"))
    recipe = output / "verify-request.json"
    recipe.write_text(json.dumps(request, indent=2) + "\n")
    from woof.ocean.waves.verify import run as verify
    from woof.hex.wave_products import render
    timings = {}
    began = time.perf_counter()
    verify(recipe)
    charts = sorted((output / "verification" / "pictures").glob("*.png"))
    if cfg.get("pictures"):
        destination = Path(cfg["pictures"]) / "stations"
        destination.mkdir(parents=True, exist_ok=True)
        for chart in charts:
            shutil.copy2(chart, destination / chart.name)
    timings["verify_and_analysis_charts"] = time.perf_counter() - began
    began = time.perf_counter()
    products = render(forecast, cfg["mesh"], cfg.get("pictures", output / "pictures"),
        domain=cfg.get("domain", "global"), bounds=tuple(cfg.get("bounds", (-180,180,-89.5,89.5))),
        size=tuple(cfg.get("size", (1200,750))), table_path=cfg.get("display_table"))
    timings["wave_maps"] = time.perf_counter() - began
    nwet, calls = wave["core"]["nwet"], wave["calls"]
    result = dict(status="complete", atmosphere_route=route, config=cfg,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        prepared_manifest_sha256=hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest(),
        forecast_frames=len(frames), start_epoch=start, end_epoch=end,
        atmosphere_steps=atmosphere_steps,
        timings_seconds=timings, wave=wave,
        wave_ms_per_atmosphere_step=wave["solver_device_ms"] / atmosphere_steps,
        wave_ms_per_call=wave["solver_device_ms"] / calls,
        ms_per_wave_step_per_million_wet_cells=wave["solver_device_ms"] / calls * 1e6 / nwet,
        wave_cost_scope=wave.get("timing_collection", "CUDA event spans; shared-card stalls included"),
        native_field_bytes=sum(p.stat().st_size for p in frames),
        map_count=len(products["maps"]),
        chart_count=len(charts),
        visual_review="PENDING: every generated map and analysis chart must be opened before qualification")
    (output / "finalize-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ("status", "forecast_frames", "map_count", "timings_seconds", "visual_review")}))


if __name__ == "__main__":
    main()
