"""Non-map analysis charts. Rust has already paired and reduced every value."""
from datetime import datetime, timezone
import json
from pathlib import Path

def _station_job(job):
    station, request, output = job
    return render(None, output, _document={"stations": [station], "request": request}, _workers=False)

def render(source, output, *, _document=None, _workers=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from matplotlib.colors import PowerNorm
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document = _document if _document is not None else json.loads(Path(source).read_text())
    pictures = []
    def save(fig, name, description):
        path = output / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pictures.append({"path": str(path), "description": description, "inspection": "pending"})
    instant = lambda v: datetime.fromtimestamp(v, timezone.utc)
    if _workers and document["stations"]:
        from concurrent.futures import ProcessPoolExecutor
        import os
        workers = min(len(document["stations"]), os.cpu_count() or 1, document["request"].get("chart_workers", 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for made in pool.map(_station_job, [(s, document["request"], output) for s in document["stations"]]):
                pictures.extend(made)
    for station in ([] if _workers else document["stations"]):
        sid = station["station"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for ax, key, ylabel in zip(axes, ("hs", "tp", "dp"), ("Hs (m)", "Tp (s)", "Dp FROM (degree)")):
            ax.plot([instant(o["time"]) for o in station["observations"]],
                    [o.get(key) for o in station["observations"]], color="black", marker=".", markersize=3, linewidth=1, linestyle="none" if key == "dp" else "-", label="NDBC")
            for series in station["series"]:
                if key in series["fields"]:
                    # Direction is scatter so a north crossing cannot imply a
                    # physically meaningful line through south.
                    args = {"linestyle": "none", "marker": "."} if key == "dp" else {"linewidth": 1.4}
                    ax.plot([instant(t) for t in series["times"]], series["fields"][key], label=series["label"], **args)
            if key == "dp": ax.set_ylim(0, 360)
            else: ax.set_ylim(bottom=0)
            ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
        axes[0].legend(ncol=3, frameon=False)
        axes[0].set_title(f"Wave evolution at buoy {sid}")
        axes[-1].xaxis.set_major_formatter(dates.DateFormatter("%m-%d\n%HZ", tz=timezone.utc))
        fig.tight_layout(); save(fig, f"{sid}-timeseries.png", "Hs, peak period and peak FROM direction; UTC")
        spectral = station.get("spectra")
        if spectral:
            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
            for ax, key, unit in zip(axes, ("energy", "mean_direction", "spread"), ("Variance density (m2/Hz)", "Mean FROM (degree)", "Spread (degree)")):
                if spectral.get(key) is not None:
                    ax.plot(spectral["frequency"], spectral[key], color="black", marker=".", linestyle="none" if key == "mean_direction" else "-", label="NDBC")
                for model in spectral["models"]:
                    ax.plot(model["frequency"], model[key], label=model["label"], **({"linestyle":"none","marker":".","markersize":3} if key == "mean_direction" else {}))
                if key == "mean_direction": ax.set_ylim(0, 360)
                ax.grid(alpha=0.25); ax.set_ylabel(unit)
            axes[0].legend(frameon=False); axes[0].set_title(f"Buoy {sid}, {instant(spectral['time']):%Y-%m-%d %H:%MZ}\nObserved Hs peak: {instant(spectral.get('observed_hs_peak_time', spectral['time'])):%m-%d %H:%MZ}")
            axes[-1].set_xlabel("Frequency (Hz)"); axes[-1].set_xlim(0.03, 0.5)
            fig.tight_layout(); save(fig, f"{sid}-frequency-spectrum.png", "Rust frequency densities, mean directions and spreading")
            panels = []
            if spectral.get("efth") is not None:
                panels.append({**spectral, "label": "NDBC MEM"})
            panels.extend(spectral["models"])
            if panels:
                fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), subplot_kw={"projection": "polar"}, squeeze=False)
                # Array reshape and angular unit conversion are only plotting
                # coordinates. Scientific spectra and all scores came from Rust.
                for ax, panel in zip(axes[0], panels):
                    values = np.asarray(panel["efth"], dtype=float).reshape(len(panel["frequency"]), len(panel["direction_from"]))
                    angle = np.deg2rad(panel["direction_from"])
                    order = np.argsort(angle)
                    mesh = ax.pcolormesh(angle[order], panel["frequency"], values[:, order], shading="nearest", cmap="magma", norm=PowerNorm(0.5, vmin=0, vmax=spectral["color_max"]))
                    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0.03, 0.5)
                    ax.set_title(f"{panel['label']}\n{instant(panel['time']):%m-%d %H:%MZ}"); fig.colorbar(mesh, ax=ax, shrink=0.7, label="m2/Hz/rad (sqrt colour spacing)")
                fig.suptitle(f"Buoy {sid}: directions waves come FROM\nObserved Hs peak: {instant(spectral.get('observed_hs_peak_time', spectral['time'])):%m-%d %H:%MZ}")
                fig.tight_layout(); save(fig, f"{sid}-directional-spectrum.png", "NDBC MEM and model spectra; shared square-root colour spacing exposes weak lobes without clipping peaks")
    if not _workers:
        return pictures
    labels = [m["label"] for m in document["request"]["models"]]
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        for station in document["stations"]:
            pairs = [p for p in station["pairs"] if p["model"] == label and p["metric"] == "hs"]
            ax.scatter([p["obs"] for p in pairs], [p["prediction"] for p in pairs], s=9, alpha=0.6, label=station["station"])
        limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, limit], [0, limit], color="black", linewidth=1)
        ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_aspect("equal")
        ax.set_xlabel("NDBC Hs (m)"); ax.set_ylabel(f"{label} Hs (m)"); ax.grid(alpha=0.25); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, "pooled-hs-scatter.png", "Common-pair significant wave heights by buoy")
    (output / "picture-manifest.json").write_text(json.dumps(pictures, indent=2) + "\n")
    return pictures
