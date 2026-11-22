# SPDX-License-Identifier: Apache-2.0
"""WOOF-ArWen as an Earth2Studio prognostic model.

WOOF-ArWen is a CUDA port of WRF-ARW: a regional physics model. As an Earth2Studio
``PrognosticModel`` it advances a state on a regular 0.25 degree latitude-
longitude box by one boundary interval (6 h for the packaged Earth-2
profile) per step:

1. the input state ``x`` (Earth2Studio variable ids: z/t/u/v/q on 13 pressure
   levels plus t2m, u10m, v10m, msl, the variables woof's
   ``earth2-pl13-q-netcdf-v1`` profile reads) is written with Earth2Studio's
   own ``NetCDF4Backend`` as lead 0 of a file, and the boundary source's
   state one interval later as the next lead;
2. woof prepares and runs WOOF-ArWen over the box (shrunk by ``margin_deg``) from
   that file, with the ERA5 analysis at the initialization supplying the land
   surface these variables lack (``woof fetch --source era5``, keyless);
3. ``woof export --earth2`` writes WOOF-ArWen's history in the same vocabulary on
   the same lattice, and the step returns it.

Inside the WOOF-ArWen domain the output is WOOF-ArWen's forecast. The ring between the
domain and the box edge is the boundary source's state at the output time,
the state WOOF-ArWen's lateral boundary is relaxed to, so the box is complete and
the next step can start from it. Every step is a full WOOF-ArWen run started from
the state it is handed (Earth2Studio's convention: the tensor is the state).

The package names no model and no case: variables, levels and the boundary
interval are read from woof's Earth-2 table at construction.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.io import NetCDF4Backend
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

from . import _gpuwm
from .data import WoofHistory

PROFILE = "earth2-pl13-q-netcdf-v1"


def _lattice(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step))
    return np.round(start + step * np.arange(n + 1), 6)


class ArWen(torch.nn.Module, PrognosticMixin):
    """WOOF-ArWen (woof) regional physics forecast as an Earth2Studio prognostic model.

    Parameters
    ----------
    area : (south, west, north, east)
        The box in degrees; longitudes in 0..360 or -180..180. Snapped to the
        0.25 degree lattice Earth2Studio's global sources use.
    boundary : DataSource
        Any Earth2Studio data source that serves the profile's variables at
        the output valid times (e.g. ``earth2studio.data.ARCO()`` for ERA5,
        ``GFS()``), or a WoofHistory / another model's output.
    dx_km : float
        WOOF-ArWen's grid spacing.
    margin_deg : float
        Ring between the box edge and the WOOF-ArWen domain, filled from ``boundary``.
    workdir : path, optional
        Where each step's run folder goes (kept for inspection when ``keep``).
    sim_args : list[str]
        Extra ``woof sim`` arguments (e.g. ``["--render-products", "all"]``).
    domain_args : list[str]
        Extra ``woof domain`` arguments (e.g. ``["--vram-gib", "8"]`` to size
        the domain for the card share the run is admitted with).
    """

    def __init__(
        self,
        area: tuple[float, float, float, float],
        boundary: DataSource,
        dx_km: float = 3.0,
        margin_deg: float = 1.0,
        profile: str = PROFILE,
        workdir: str | Path | None = None,
        keep: bool = True,
        sim_args: list[str] | None = None,
        domain_args: list[str] | None = None,
        grid_deg: float = 0.25,
    ):
        super().__init__()
        south, west, north, east = (float(v) for v in area)
        west %= 360.0
        east %= 360.0
        if east <= west:
            raise ValueError("area must not cross the 0 meridian in 0..360 longitude "
                             f"(west {west}, east {east})")
        g = grid_deg
        self.lat = _lattice(np.ceil(north / g) * g, np.floor(south / g) * g, -g)
        self.lon = _lattice(np.floor(west / g) * g, np.ceil(east / g) * g, g)
        self.margin = float(margin_deg)
        if 2 * self.margin >= min(self.lat[0] - self.lat[-1], self.lon[-1] - self.lon[0]):
            raise ValueError("margin_deg leaves no WOOF-ArWen domain inside the box")
        self.boundary = boundary
        self.dx_km = float(dx_km)
        self.profile = profile
        self.variables = _gpuwm.profile_variables(profile)
        self.dt = np.timedelta64(_gpuwm.profile_boundary_hours(profile), "h")
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="arwen-px-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.sim_args = list(sim_args or [])
        self.domain_args = list(domain_args or [])
        self.receipts: list[dict] = []
        self.register_buffer("_device_probe", torch.empty(0))

    def __str__(self) -> str:
        return "arwen"

    # ---- coordinates ---------------------------------------------------------
    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "lat": self.lat.copy(),
                "lon": self.lon.copy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        out = self.input_coords()
        out["lead_time"] = np.array([self.dt])
        if input_coords is None:
            return out
        test = input_coords.copy()
        test["lead_time"] = test["lead_time"] - input_coords["lead_time"][-1]
        target = self.input_coords()
        for i, key in enumerate(target):
            if key not in ("batch", "time"):
                handshake_dim(test, key, i)
                handshake_coords(test, target, key)
        out["batch"] = input_coords["batch"]
        out["time"] = input_coords["time"]
        out["lead_time"] = out["lead_time"] + input_coords["lead_time"]
        return out

    # ---- one ArWen run -------------------------------------------------------
    def _box_polygon(self, inset: float) -> dict:
        s, n = self.lat[-1] + inset, self.lat[0] - inset
        w, e = self.lon[0] + inset, self.lon[-1] - inset
        if s >= n or w >= e:
            raise ValueError("the WOOF-ArWen domain this box needs does not fit inside it; "
                             "give a larger area or a smaller margin_deg")
        w180 = ((w + 180.0) % 360.0) - 180.0
        e180 = ((e + 180.0) % 360.0) - 180.0
        ring = [[w180, s], [e180, s], [e180, n], [w180, n], [w180, s]]
        return {"type": "Polygon", "coordinates": [ring]}

    def _boundary_state(self, valid: datetime) -> np.ndarray:
        da = self.boundary([valid], self.variables)
        da = da.sel(lat=self.lat, lon=self.lon, method="nearest", tolerance=1e-3)
        return da.isel(time=0).transpose("variable", "lat", "lon").values.astype(np.float32)

    def _write_input(self, path: Path, init: datetime, state0: np.ndarray, state1: np.ndarray) -> None:
        coords = OrderedDict(
            {
                "time": np.array([np.datetime64(init, "ns")]),
                "lead_time": np.array([np.timedelta64(0, "h"), self.dt]),
                "lat": self.lat,
                "lon": self.lon,
            }
        )
        io = NetCDF4Backend(str(path), backend_kwargs={"mode": "w"})
        io.add_array(coords, self.variables)
        both = np.stack([state0, state1], axis=1)  # variable, lead, lat, lon
        tensor = torch.from_numpy(both[:, None])  # variable, time, lead, lat, lon
        io.write([tensor[i] for i in range(len(self.variables))], coords, self.variables)
        io.close()

    def _run_one(self, init: datetime, state0: np.ndarray) -> np.ndarray:
        hours = int(self.dt / np.timedelta64(1, "h"))
        stamp = init.strftime("%Y%m%dT%H")
        run = self.workdir / f"arwen-{stamp}"
        valid = init + timedelta(hours=hours)
        state1 = self._boundary_state(valid)
        # A finished ArWen run is reused when its inputs are byte-identical
        # (same state, boundary and settings): the forecast is deterministic
        # in them, so only the export and later stages run again.
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(state0, np.float32).tobytes())
        digest.update(np.ascontiguousarray(state1, np.float32).tobytes())
        digest.update(json.dumps([init.isoformat(), hours, self.dx_km, self.margin, self.profile,
                                  self.domain_args, self.sim_args,
                                  self.lat.tolist(), self.lon.tolist()]).encode())
        key = digest.hexdigest()
        done = run / "forecast-done.json"
        reuse = (done.exists() and json.loads(done.read_text()).get("inputs_sha256") == key)
        if run.exists() and not reuse:
            shutil.rmtree(run)
        run.mkdir(parents=True, exist_ok=True)
        log = run / "gpuwm.log"
        receipt: dict = {"init": init.isoformat(), "hours": hours, "dx_km": self.dx_km,
                         "profile": self.profile, "inputs_sha256": key, "stages_s": {}}
        if reuse:
            receipt = json.loads(done.read_text())
            receipt["reused_forecast"] = True
            return self._finish(run, receipt, valid, state1)
        src = run / f"earth2-arwen-{stamp}.nc"
        self._write_input(src, init, state0, state1)
        s, w, n, e = self.lat[-1], self.lon[0], self.lat[0], self.lon[-1]
        area = f"{s},{((w + 180) % 360) - 180},{n},{((e + 180) % 360) - 180}"
        cyc = init.strftime("%Y-%m-%dT%H")
        st = receipt["stages_s"]
        st["era5_surface"] = _gpuwm.run(
            ["fetch", "--source", "era5", "--era5-provider", "arco", "--retrieve",
             "--cycle", cyc, "--hours", "0", "--area", area, "--out", "analysis"],
            cwd=run, log=log)
        # The domain woof fits around the polygon reads a wider window of the
        # input than the polygon (projection rectangle plus interpolation
        # halo); woof prints that window.  Shrink the polygon until the
        # window lies inside the box the input file covers.
        inset, st["domain"] = self.margin, 0.0
        for _attempt in range(4):
            (run / "box.geojson").write_text(json.dumps(self._box_polygon(inset)), encoding="utf-8")
            mark = log.stat().st_size
            st["domain"] += _gpuwm.run(
                ["domain", "--source", "earth2", "--polygon", "box.geojson",
                 "--root-dx", str(self.dx_km), "--hours", str(hours), "--cycle", cyc,
                 "--out", "px.toml", *self.domain_args], cwd=run, log=log)
            text = " ".join(log.read_text(encoding="utf-8", errors="replace")[mark:]
                            .replace("#", " ").split())
            m = re.search(r"window this config needs is (-?[0-9.]+),(-?[0-9.]+),(-?[0-9.]+),(-?[0-9.]+)", text)
            if not m:
                break
            ws, ww, wn, we = (float(v) for v in m.groups())
            bs, bn = float(self.lat[-1]), float(self.lat[0])
            bw = ((float(self.lon[0]) + 180) % 360) - 180
            be = ((float(self.lon[-1]) + 180) % 360) - 180
            over = max(bs - ws, wn - bn, bw - ww, we - be)
            receipt["domain_window"] = [ws, ww, wn, we]
            receipt["domain_inset_deg"] = inset
            if over <= 0:
                break
            inset += over + 0.25
        else:
            raise RuntimeError(f"no WOOF-ArWen domain inside the box after 4 fits; last window {receipt.get('domain_window')}")
        st["prep"] = _gpuwm.run(
            ["prep", "--source", "earth2", "--input", src.name,
             "--supplement", "physical_analysis_surface_data=analysis/era5-combined.nc",
             "--author-input-manifest", "inputs.json",
             "--geog-root", _gpuwm.default_geog_root(),
             "--experiment-config", "px.toml", "--wps-namelist", "px.namelist.wps",
             "--output-root", "prepared"], cwd=run, log=log)
        st["sim"] = _gpuwm.run(
            ["sim", "prepared", "--experiment-config", "px.toml",
             "--wps-namelist", "px.namelist.wps", "--outdir", "forecast",
             "--run-stamp", "off", *self.sim_args],
            cwd=run, log=log)
        done.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        return self._finish(run, receipt, valid, state1)

    def _finish(self, run: Path, receipt: dict, valid: datetime, state1: np.ndarray) -> np.ndarray:
        st = receipt["stages_s"]
        hist = WoofHistory(run / "forecast", cache_dir=run / "earth2-export",
                           profile="e2s-latlon-0.25", channels=self.variables)
        arwen = hist([valid], self.variables)
        st["export"] = hist.export_seconds
        arwen = arwen.reindex(lat=self.lat, lon=self.lon, method="nearest", tolerance=1e-3)
        out = arwen.isel(time=0).transpose("variable", "lat", "lon").values.astype(np.float32)
        inside = np.isfinite(out)
        receipt["arwen_fraction_of_box"] = float(inside.all(axis=0).mean())
        out = np.where(inside, out, state1)
        receipt["output_valid"] = valid.isoformat()
        (run / "px-receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        self.receipts.append(receipt)
        if not self.keep:
            shutil.rmtree(run / "prepared", ignore_errors=True)
        return out

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
        out_coords = self.output_coords(coords)
        y = torch.empty((x.shape[0], x.shape[1], 1, *x.shape[3:]), dtype=x.dtype, device=x.device)
        for b in range(x.shape[0]):
            for i, t in enumerate(coords["time"]):
                init = pd.Timestamp(t + coords["lead_time"][-1]).to_pydatetime()
                state0 = x[b, i, -1].detach().float().cpu().numpy()
                y[b, i, 0] = torch.from_numpy(self._run_one(init, state0)).to(x.device, x.dtype)
        return y, out_coords

    @batch_func()
    def __call__(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()
        self.output_coords(coords)
        yield x, coords
        while True:
            x, coords = self.front_hook(x, coords)
            x, coords = self._forward(x, coords)
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        yield from self._default_generator(x, coords)
