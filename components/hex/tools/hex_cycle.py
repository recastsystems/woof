"""One global hex forecast cycle, GFS on AWS to maps, every stage on one timeline.

    python tools/hex_cycle.py run --work DIR --mesh-dir DIR --mesh-rows FILE \
        --layouts NPZ --part FILE [--cycle newest|YYYYMMDDHH] [--wait-seconds S] \
        [--hours 24] [--history-minutes 180] [--dt 90] [--radiation 1800] \
        [--products LIST] [--render-workers 2] [--label hex_x1.2621442] \
        [--forecast-launcher "ARGV..."]

Stages, in order, each a begin/end pair in DIR/cycle-timeline.jsonl
(epoch seconds and UTC, the same clock the forecast's own timeline writes):

  cycle.available  the S3 Last-Modified of the cycle's f000 (a mark)
  fetch            rw_fetch: parallel range GETs of gfs.tHHz.pgrb2.0p25.f000
  intermediate     met_intermediate (Rust) with the pinned Vtable.GFS
  init             hexcore init straight onto the ORDERED grid and static,
                   with the constructed vertical artifact taken from the
                   vertical cache (the ordering, partition and vertical
                   artifact are mesh constants, built once)
  forecast         the 8-rank same-host forecast; its history frames land
                   off the step path and are announced in
                   fc/history-ready.jsonl
  render.frame     one per history frame, started the moment that frame is
                   announced, on --render-workers CPU workers, through the
                   Rust render door at the mesh's own spacing
  cycle            everything, from the start of fetch to the last map

Why the init goes straight onto the ordered mesh: init-then-reorder wrote
the 24.6 GB init twice (8-card RTX PRO 6000 host, 2026-10-10: hex_order --apply 98 s plus
--record-donor 11 s); an init built on the ordered grid and static has the
same 138 variables byte for byte (compared by name on the 8-card RTX PRO 6000 host, 2026-10-10), the
ordering metadata carried from the ordered static, and only a different
variable order in the header.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from woof.hex import card_queue
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
DEFAULT_PRODUCTS = ",".join((
    "500mb_height_winds", "250mb_height_winds", "850mb_temperature_height_winds", "2m_temperature",
    "mslp_10m_winds", "total_qpf", "precipitable_water", "composite_reflectivity",
    "10m_wind_speed_and_direction", "2m_dewpoint",
))
INIT_SWITCHES = (
    "--nfglevels", "34", "--nfgsoillevels", "4", "--extrap-airtemp", "constant", "--use-spechumd", "no",
    "--theta-adv-order", "3", "--coef-3rd-order", "0.25", "--virtual-factor", "reproduce-fortran",
    "--deep-soil-moisture", "reproduce-fortran", "--landuse-table", "MODIFIED_IGBP_MODIS_NOAH",
    "--frac-seaice", "yes", "--tsk-seaice-threshold", "100", "--oned-underflow", "preserve",
)


class Timeline:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def mark(self, stage: str, phase: str = "mark", **attrs):
        now = time.time()
        row = {"stage": stage, "phase": phase, "u": now,
               "utc": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"), **attrs}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def run(self, stage: str, command: list[str], log: Path, env=None, **attrs) -> None:
        self.mark(stage, "begin", **attrs)
        started = time.time()
        with log.open("w") as stream:
            result = subprocess.run([str(part) for part in command], stdout=stream, stderr=subprocess.STDOUT,
                                    env=env)
        self.mark(stage, "end", rc=result.returncode, seconds=round(time.time() - started, 3), **attrs)
        if result.returncode != 0:
            raise SystemExit(f"{stage} failed rc={result.returncode}; see {log}")


def _head(url: str):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as response:
            return parsedate_to_datetime(response.headers["Last-Modified"])
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return None
        raise


def resolve_cycle(text: str, rw_fetch: Path) -> str:
    if text != "newest":
        return text
    result = subprocess.run([str(rw_fetch), "latest", "--model", "gfs", "--product", "pgrb2.0p25",
                             "--through", "0", "--source", "aws"], capture_output=True, text=True, check=True)
    found = json.loads(result.stdout)["cycle"]  # gpuwm-rw-fetch-latest-v1
    return f"{found['date']}{int(found['hour']):02d}"


def render_frame(args, ready: dict, timeline: Timeline, init: Path, start: str) -> None:
    step = int(ready["step"])
    history = Path(ready["path"])
    scratch = (args.render_scratch_root or args.work / "render-scratch") / f"step{step:06d}"
    command = ["nice", "-n", "5", sys.executable, "-m", "woof.hex.cli", "render", "--history", str(history),
               "--mesh", str(args.mesh_dir / "ordered.static.nc"), "--init", str(init),
               "--out", str(args.work / "maps"), "--scratch", str(scratch), "--products", args.products,
               "--window", "mesh", "--size", args.size, "--simulation-start", start, "--nc-format", "cdf5",
               "--label", args.label]
    if args.convert_exe:
        command += ["--convert-exe", str(args.convert_exe)]
    if args.renderer_exe:
        command += ["--renderer-exe", str(args.renderer_exe)]
    (args.work / "logs").mkdir(exist_ok=True)
    try:
        timeline.run("render.frame", command, args.work / "logs" / f"render-step{step:06d}.log",
                     step=step, history=history.name)
    except SystemExit as error:
        timeline.mark("render.frame", "failed", step=step, error=str(error))


def watch_history(args, fc: Path, timeline: Timeline, init: Path, start: str, stop: threading.Event,
                  pool: ThreadPoolExecutor, futures: list) -> None:
    ready_file = fc / "history-ready.jsonl"
    seen = 0
    while True:
        rows = []
        if ready_file.exists():
            rows = [json.loads(line) for line in ready_file.read_text().splitlines() if line.strip()]
        for row in rows[seen:]:
            timeline.mark("history.ready", step=row["step"], path=Path(row["path"]).name,
                          writer_seconds=row.get("seconds"))
            futures.append(pool.submit(render_frame, args, row, timeline, init, start))
        seen = len(rows)
        if stop.is_set():
            # one last look after the forecast exits: every frame is announced by then
            if not ready_file.exists() or len(ready_file.read_text().splitlines()) == seen:
                return
        time.sleep(1.0)


def run(args) -> int:
    args.work.mkdir(parents=True, exist_ok=True)
    for sub in ("gfs", "met", "init", "logs"):
        (args.work / sub).mkdir(exist_ok=True)
    timeline = Timeline(args.work / "cycle-timeline.jsonl")
    rw_fetch = Path(args.rw_fetch or Path(os.environ["RWBIN"]) / "rw_fetch")
    met_intermediate = Path(args.met_intermediate or os.environ["MET_INTERMEDIATE"])
    tree = Path(__file__).resolve().parents[1]
    cycle = resolve_cycle(args.cycle, rw_fetch)
    day, hh = cycle[:8], cycle[8:10]
    stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}_{hh}"
    start = f"{stamp}:00:00"
    url = f"{BUCKET}/gfs.{day}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000"
    init = args.work / "init" / "ordered.init.nc"
    if args.resume_at == "forecast":
        # The front half already ran into this work directory (hex_cycle --until init) while the box's
        # cards were busy; the gap until now is a card-queue wait, marked so a report can cut it out.
        if not (init.is_file() and Path(str(init) + ".provenance.json").is_file()):
            raise SystemExit(f"--resume-at forecast needs the front half's init at {init}")
        timeline.mark("forecast.grant", cards=card_queue.get(card_queue.CARDS))
    else:
        timeline.mark("cycle", "begin", cycle=cycle)
        available = _head(url)
        waited = time.time()
        while available is None and time.time() - waited < args.wait_seconds:
            time.sleep(10)
            available = _head(url)
        if available is None:
            raise SystemExit(f"{url} is not available")
        timeline.mark("cycle.available", available_utc=available.isoformat(), available_u=available.timestamp(),
                      url=url)
        timeline.run("fetch", [rw_fetch, "fetch", "--model", "gfs", "--date", day, "--cycle", hh, "--hours", "0",
                               "--product", "pgrb2.0p25", "--out", args.work / "gfs", "--source", "aws",
                               "--mode", "full-file"], args.work / "logs" / "fetch.log")
        grib = args.work / "gfs" / f"gfs.t{hh}z.pgrb2.0p25.f000"
        met = args.work / "met" / f"MET:{stamp}"
        timeline.run("intermediate", [met_intermediate, "--vtable", args.vtable, "--date", f"{stamp}:00:00",
                                      "--map-source", "ncep-gfs", "--out", met, grib],
                     args.work / "logs" / "intermediate.log")
        timeline.run("init", [sys.executable, "-m", "woof.hex.cli", "init", "--met", met,
                              "--grid", args.mesh_dir / "ordered.grid.nc",
                              "--static", args.mesh_dir / "ordered.static.nc",
                              "--vertical-spec", tree / "verification/vertical-specs/tc55-v1.json", "--out", init,
                              "--start-time", start, *INIT_SWITCHES], args.work / "logs" / "init.log")
        if args.until == "init":
            timeline.mark("front_half", "end", cycle=cycle)
            return 0

    steps = int(round(args.hours * 3600 / args.dt))
    fc = args.work / "fc"
    forecast = [*shlex.split(args.forecast_launcher), "forecast"] if args.forecast_launcher else \
        [sys.executable, "-m", "woof.hex.cli", "forecast"]
    forecast += ["--mesh", args.mesh, "--grid", args.mesh_dir / "ordered.grid.nc",
                 "--static", args.mesh_dir / "ordered.static.nc", "--init", init, "--hours", f"{args.hours:g}",
                 "--start-time", start, "--history-every-minutes", str(args.history_minutes),
                 "--init-source", f"GFS {day} {hh}Z 0.25 deg (AWS); rw_fetch, met_intermediate, hexcore init on "
                                  f"the ordered mesh", "--out", fc, "--scratch", args.work / "scratch",
                 "--case-label", "hex-cycle", "--gpuwm-checkout", os.environ["E"],
                 "--samehost-devices", args.devices, "--samehost-layouts", args.layouts,
                 "--samehost-part", args.part, "--samehost-fingerprint-every", str(steps), "--stop-on-refusal"]
    if args.radiation:
        forecast += ["--radiation-seconds", f"{args.radiation:g}"]
    forecast += shlex.split(args.forecast_extra)
    env = dict(os.environ, WOOF_HEX_MESH_ROWS=str(args.mesh_rows), GPUWM_WRF_EXACT="0",
               WOOF_HEX_REGIONAL_CONTRACT_DIR=str(args.work / "contracts"))
    stop = threading.Event()
    futures: list = []
    with ThreadPoolExecutor(max_workers=args.render_workers, thread_name_prefix="render") as pool:
        watcher = threading.Thread(target=watch_history, name="history-watch",
                                   args=(args, fc, timeline, init, start, stop, pool, futures), daemon=True)
        watcher.start()
        try:
            timeline.run("forecast", forecast, args.work / "forecast.log", env=env, steps=steps)
        finally:
            stop.set()
            watcher.join()
        for future in list(futures):
            future.result()
    timeline.mark("cycle", "end", cycle=cycle)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--mesh-dir", type=Path, required=True, help="holds ordered.grid.nc and ordered.static.nc")
    r.add_argument("--mesh", required=True, help="registered mesh row name, e.g. x1.2621442-ordered")
    r.add_argument("--mesh-rows", type=Path, required=True)
    r.add_argument("--layouts", type=Path, required=True)
    r.add_argument("--part", type=Path, required=True)
    r.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    r.add_argument("--cycle", default="newest")
    r.add_argument("--wait-seconds", type=float, default=0.0)
    r.add_argument("--hours", type=float, default=24.0)
    r.add_argument("--history-minutes", type=int, default=180)
    r.add_argument("--dt", type=float, default=90.0)
    r.add_argument("--radiation", type=float, default=1800.0)
    r.add_argument("--products", default=DEFAULT_PRODUCTS)
    r.add_argument("--size", default="2700x1500")
    r.add_argument("--label", default="hex")
    r.add_argument("--render-workers", type=int, default=2)
    r.add_argument("--render-scratch-root", type=Path, default=None,
                   help="per-frame converter scratch (default WORK/render-scratch); a tmpfs keeps the "
                        "13.6 GB native-resolution frame of a 15 km mesh off the disk")
    r.add_argument("--vtable", type=Path, required=True, help="WPS Vtable.GFS the intermediate is decoded with")
    r.add_argument("--rw-fetch", type=Path, default=None)
    r.add_argument("--met-intermediate", type=Path, default=None)
    r.add_argument("--convert-exe", type=Path, default=None)
    r.add_argument("--renderer-exe", type=Path, default=None)
    r.add_argument("--resume-at", choices=("start", "forecast"), default="start",
                   help="forecast: the front half already ran into --work (--until init); start at the forecast")
    r.add_argument("--until", choices=("init", "maps"), default="maps",
                   help="stop after this stage (init: the CPU front half only, no card needed)")
    r.add_argument("--forecast-extra", default="",
                   help="more forecast-door flags, e.g. a schedule other than the MPAS-A defaults")
    r.add_argument("--forecast-launcher", default="",
                   help="argv prefix that ends where the forecast subcommand begins (default: woof.hex.cli)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
