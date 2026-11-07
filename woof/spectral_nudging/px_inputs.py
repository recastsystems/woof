"""Resident native auxinput9 endpoints for PX's independent soil nudging.

Rust reads stored words once. CUDA selects OLD/NEW records on every domain
step; PX performs its native humidity and time interpolation in its column.
The clock is rebuilt from checkpoint elapsed time, independently per member.
"""
from datetime import datetime
from pathlib import Path
import numpy as np


class PxInputs:
    def __init__(self, state, cfg, path, start_time, run_seconds):
        import cupy as cp
        from woof.netcdf_bridge import Dataset
        from woof.file_digest import sha256_file
        from .surface_inputs import read_native_surface
        self.state, self.cfg, self.path = state, cfg, Path(path)
        self.sha256 = sha256_file(self.path)
        interval = cfg.sgfdda_interval_m * 60.
        with Dataset(path) as dataset:
            times = dataset.stored_words(("Times",))["Times"]
        offsets = [(datetime.fromisoformat(row.tobytes().decode("ascii").rstrip("\0 ").replace("_", "T"))
                    - start_time.replace(tzinfo=None)).total_seconds() for row in times]
        if cfg.use_adaptive_time_step:
            raise ValueError("PX_ANALYSIS_ADAPTIVE_CLOCK: native analysis clocks require fixed model steps")
        if (not offsets or offsets[0] != 0 or
                any(abs(b-a-interval)>1.e-6 for a,b in zip(offsets, offsets[1:])) or
                offsets[-1]+interval < run_seconds):
            raise ValueError("PX_ANALYSIS_COVERAGE: native auxinput9 OLD/NEW records must bracket the whole soil-nudging run")
        selected = [i for i,t in enumerate(offsets) if t < max(run_seconds, 1.)]
        self.names = tuple(f"{name}2_ndg_{end}" for name in ("t", "q") for end in ("old", "new")) + ("sn_ndg_old", "sn_ndg_new")
        self.records = cp.empty((len(selected), 6, cfg.ny, cfg.nx), cp.float32)
        for out, record in enumerate(selected):
            fields = read_native_surface(path, record=record, surface=False, px=True).fields
            for n,name in enumerate(self.names):
                value = fields[name.upper()]
                if value.shape != (cfg.ny, cfg.nx):
                    raise ValueError(f"PX native {name} shape differs from its domain")
                self.records[out,n] = cp.asarray(value)
        self.offsets = cp.asarray([offsets[i] for i in selected], dtype=cp.float64)
        self.clock = cp.asarray([state.elapsed_seconds, state.elapsed_seconds], dtype=cp.float64)
        self.current = self.clock[1:]
        source = Path(__file__).with_suffix(".cu").read_text()
        self.module = cp.RawModule(code=source, options=("--std=c++17",))
        self.kernel = self.module.get_function("px_select")
        self.advance = self.module.get_function("px_advance")
        from woof.certify.kernel_manifest import record_module
        record_module("woof.spectral_nudging.px_inputs",source=source,options=("--std=c++17",),module=self.module)
        f = state.physics.fields
        self.args = (self.clock, self.offsets, self.records,
                     *(f[name] for name in self.names), np.int32(len(selected)),
                     np.int32(cfg.ny*cfg.nx), np.float64(cfg.dt))
        self.grid = ((cfg.ny*cfg.nx+127)//128,)
        state._px_inputs_owner = self

    def before_step(self):
        # Clock update and copies share one kernel; every block calculates
        # its time from the same pre-step counter. A second kernel advances
        # the counter after every reader, avoiding any inter-block race.
        self.kernel(self.grid, (128,), self.args)
        self.advance((1,), (1,), (self.clock, np.float64(self.cfg.dt)))

    def check_at_seam(self):
        if not np.isfinite(self.clock.get()).all():
            raise RuntimeError("PX analysis clock became nonfinite")

    def receipt(self):
        return dict(input=str(self.path), input_sha256=self.sha256,
                    records=len(self.offsets), data_path="Rust stored words; CUDA device clock and record selection")

    def detach(self):
        self.state._px_inputs_owner = None
