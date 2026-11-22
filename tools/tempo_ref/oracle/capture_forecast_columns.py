"""Capture the TEMPO column inputs of chosen calls of an ordinary forecast.

    export WOOF_TEMPO_CAPTURE=DIR WOOF_TEMPO_CAPTURE_CALLS=360,720
    export PYTHONPATH=<this directory>/capture_site:$PYTHONPATH
    woof go case.toml --outdir RUN ...

``capture_site/sitecustomize.py`` calls :func:`install_lazy` in every Python
process of the run (``woof go`` runs its stages as subprocesses), so the
forecast is the ordinary product run.  Before the listed microphysics calls
(1-based, counted per process) the column state ``TempoRuntime.step``
receives is written, before the call changes it, as
``DIR/call<N>_columns_in.bin`` plus ``call<N>_columns_catalog.json`` in the
column-oracle layout (make_columns.py): real forecast columns for the
Fortran referee (tempo_oracle_driver.F90) and the GPU replay
(run_woof_oracle.py).  Temperature is the column's own float64 product
theta * Exner and graupel volume is converted to L/kg in float64, exactly as
tempo_pack_adapter forms them; the call's dt is recorded with the set.
Capturing reads the fields once to the host per listed call; no other call
is touched.  Without WOOF_TEMPO_CAPTURE nothing is installed.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
          "nwfa", "nifa", "ng", "qb")
_TARGET = "woof.core.tempo_runtime"


def _patch(tempo_runtime):
    import numpy as np

    out = Path(os.environ["WOOF_TEMPO_CAPTURE"])
    calls = {int(x) for x in os.environ.get("WOOF_TEMPO_CAPTURE_CALLS", "1").split(",") if x}
    original = tempo_runtime.TempoRuntime.step
    count = {"n": 0}

    def step(self, fields, dt, cfg=tempo_runtime.TempoConfig(), *, exner=None, **kw):
        count["n"] += 1
        n = count["n"]
        if n in calls:
            out.mkdir(parents=True, exist_ok=True)
            host = {name: np.asarray(fields[name].get(), dtype=np.float64) for name in FIELDS}
            if exner is not None:
                host["t"] = host["t"] * np.asarray(exner.get(), dtype=np.float64)
                host["qb"] = host["qb"] * 1000.0
            nz, ny, nx = host["t"].shape
            ncol = ny * nx
            path = out / f"call{n}_columns_in.bin"
            with path.open("wb") as f:
                np.array([1], dtype="<i8").tofile(f)
                np.array([ncol, nz, 1], dtype="<i8").tofile(f)
                np.array([float(dt)], dtype="<f8").tofile(f)
                for name in FIELDS:
                    np.ascontiguousarray(host[name].reshape(nz, ncol).T, dtype="<f8").tofile(f)
            catalog = [dict(set=1, column=c + 1, regime="forecast",
                            variant=f"call{n}-j{c // nx}-i{c % nx}",
                            description="captured WOOF-ArWen forecast column") for c in range(ncol)]
            (out / f"call{n}_columns_catalog.json").write_text(json.dumps(catalog))
            (out / f"call{n}_meta.json").write_text(json.dumps(dict(
                call=n, dt=float(dt), shape=[nz, ny, nx], version=self.version,
                hailhyperaware=bool(cfg.hailhyperaware), first_call=bool(self.first_call))))
            print(f"[tempo-capture] call {n}: {ncol} columns x {nz} levels -> {path}", flush=True)
        return original(self, fields, dt, cfg, exner=exner, **kw)

    tempo_runtime.TempoRuntime.step = step


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)
        loader.exec_module = exec_and_patch
        return spec


def install_lazy():
    if os.environ.get("WOOF_TEMPO_CAPTURE") and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
