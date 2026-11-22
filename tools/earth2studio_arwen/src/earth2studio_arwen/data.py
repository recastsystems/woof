# SPDX-License-Identifier: Apache-2.0
"""WOOF history (WOOF-ArWen or hex) as an Earth2Studio data source.

``WoofHistory`` reads a woof / woof hex run's history through woof's own
Earth-2 exporter (``woof export --earth2``), which writes the run in
Earth2Studio's vocabulary (u10m, t2m, msl, z500, q850, ...) on a regular
latitude-longitude grid. The export happens once per run folder and is
cached beside it; later calls only read. A path that already is an export
(a ``.zarr`` store or ``.nc`` file in that vocabulary) is read as it is.

It satisfies ``earth2studio.data.DataSource``: ``source(time, variable)``
returns a DataArray with dims (time, variable, lat, lon), ``time`` being the
VALID time. Any Earth2Studio diagnostic, downscaler or tracker whose grid the
export matches can run on it.
"""
from __future__ import annotations

import asyncio
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _gpuwm



def _as_list(value) -> list:
    if isinstance(value, (str, datetime, np.datetime64, pd.Timestamp)):
        return [value]
    return list(value)


class WoofHistory:
    """A WOOF-ArWen or hex run's history as an Earth2Studio data source.

    Parameters
    ----------
    source : path or list of paths
        A run's history (a folder holding ``wrfout_d0N_*`` frames, the frames
        a hex run converts its mesh history into, or the files themselves),
        or an existing Earth-2 export (``.zarr`` / ``.nc``).
    cache_dir : path, optional
        Where the export is written (default: ``<source>/../earth2-export``).
    profile : str
        woof's Earth-2 grid profile: ``e2s-latlon-0.25`` (Earth2Studio's global
        721x1440 lattice; a regional run fills its own footprint and leaves the
        rest missing), ``e2s-healpix``, ``e2s-hrrr-3km`` or ``e2s-native``.
    consumers : str, optional
        Earth2Studio models whose inputs to export, by their row in woof's
        Earth-2 table (``windgust-afno``, ``precip-afno`` ...).
    channels : list[str], optional
        Extra Earth2Studio variable ids to export.
    export_args : list[str]
        Extra ``woof export`` arguments (``--domains d02`` ...).
    """

    def __init__(self, source, cache_dir=None, profile: str = "e2s-latlon-0.25",
                 consumers: str | None = None, channels: list[str] | None = None,
                 export_args: list[str] | None = None):
        sources = [Path(p) for p in ([source] if isinstance(source, (str, Path)) else source)]
        self.sources = sources
        self.source = sources[0]
        self.profile = profile
        self.consumers = consumers
        self.channels = list(channels or [])
        self.export_args = list(export_args or [])
        self.export_seconds = 0.0
        if self.source.suffix in (".zarr", ".nc") and self.source.exists():
            self.out_dir, self.store = None, self.source
        else:
            self.out_dir = Path(cache_dir) if cache_dir else self.source.parent / "earth2-export"
            self.store = None
        self._ds: xr.Dataset | None = None

    # ---- export (once) -------------------------------------------------------
    def _find_store(self) -> Path | None:
        if self.out_dir is None or not self.out_dir.exists():
            return None
        stores = sorted(self.out_dir.glob("*.zarr")) or sorted(self.out_dir.glob("*.nc"))
        if not stores and ((self.out_dir / "zarr.json").exists() or (self.out_dir / ".zgroup").exists()):
            return self.out_dir
        return stores[0] if stores else None

    def _ensure(self) -> xr.Dataset:
        if self._ds is not None:
            return self._ds
        if self.store is None:
            self.store = self._find_store()
        if self.store is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            args = ["export", *map(str, self.sources), "--out", str(self.out_dir),
                    "--earth2", self.profile, "--overwrite"]
            if self.consumers:
                args += ["--for", self.consumers]
            if self.channels:
                args += ["--channels", ",".join(self.channels)]
            t0 = _time.perf_counter()
            _gpuwm.run(args + self.export_args, cwd=self.out_dir.parent,
                       log=self.out_dir.parent / "earth2-export.log")
            self.export_seconds = _time.perf_counter() - t0
            self.store = self._find_store()
            if self.store is None:
                raise FileNotFoundError(f"woof export wrote no zarr/netcdf store in {self.out_dir}")
        if self.store.suffix == ".nc":
            ds = xr.open_dataset(self.store)
        else:
            ds = xr.open_zarr(self.store, consolidated=None)
        self._ds = self._valid_time_view(ds)
        return self._ds

    @staticmethod
    def _valid_time_view(ds: xr.Dataset) -> xr.Dataset:
        """One ``time`` axis of valid times, whatever layout the store has."""
        if "lead_time" in ds.dims:
            if ds.sizes.get("time", 1) != 1:
                raise ValueError("an export with several initializations needs one "
                                 "WoofHistory per initialization")
            init = pd.Timestamp(ds["time"].values[0])
            ds = ds.isel(time=0, drop=True)
            valid = np.array([np.datetime64(init + pd.Timedelta(l)) for l in ds["lead_time"].values])
            ds = ds.assign_coords(lead_time=valid).rename({"lead_time": "time"})
        if "variable" in ds.dims and len(ds.data_vars) == 1:
            ds = ds[list(ds.data_vars)[0]].to_dataset(dim="variable")
        return ds

    # ---- DataSource ----------------------------------------------------------
    @property
    def variables(self) -> list[str]:
        ds = self._ensure()
        return [v for v in ds.data_vars if {"lat", "lon"} <= set(ds[v].dims)]

    def __call__(self, time, variable) -> xr.DataArray:
        ds = self._ensure()
        times = [np.datetime64(pd.Timestamp(t).to_datetime64(), "ns") for t in _as_list(time)]
        names = [str(v) for v in _as_list(variable)]
        missing = [v for v in names if v not in ds.data_vars]
        if missing:
            raise KeyError(f"{', '.join(missing)} not in this run's Earth-2 export "
                           f"({self.store}); it has {', '.join(self.variables)}")
        have = set(ds["time"].values.astype("datetime64[ns]"))
        absent = [str(t) for t in times if t not in have]
        if absent:
            raise KeyError(f"valid times {', '.join(absent)} not in this run's history "
                           f"({str(ds['time'].values[0])} .. {str(ds['time'].values[-1])})")
        arrays = [ds[v].sel(time=times) for v in names]
        da = xr.concat(arrays, dim=pd.Index(names, name="variable"))
        return da.transpose("time", "variable", "lat", "lon").load()

    async def fetch(self, time, variable) -> xr.DataArray:
        return await asyncio.to_thread(self.__call__, time, variable)
