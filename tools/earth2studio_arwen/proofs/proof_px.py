"""hf-5 proof 2: step WOOF-ArWen as an Earth2Studio prognostic model for 6 h.

ERA5 (earth2studio ARCO, keyless) gives the initial state; the same source
gives the 6 h lateral boundary. One earth2studio.run.deterministic call.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import earth2studio.run as run
from earth2studio.data import ARCO
from earth2studio.io import NetCDF4Backend
from earth2studio.models.px.base import PrognosticModel

from earth2studio_arwen import ArWen

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
init = sys.argv[2] if len(sys.argv) > 2 else "2024-05-21T12:00"
area = (30.0, -105.0, 44.0, -89.0)  # south, west, north, east

t0 = time.perf_counter()
model = ArWen(area, boundary=ARCO(), dx_km=3.0, margin_deg=2.0, workdir=out / "runs",
              sim_args=["--render-products", "all"],
              domain_args=["--vram-gib", "12"])
assert isinstance(model, PrognosticModel)
io = NetCDF4Backend(str(out / "arwen_px_6h.nc"), backend_kwargs={"mode": "w"})
run.deterministic([np.datetime64(init)], 1, model, ARCO(), io, device="cuda")
io.close()
wall = time.perf_counter() - t0
receipt = {"proof": "earth2studio.run.deterministic with earth2studio_arwen.WOOF-ArWen, 1 step",
           "init": init, "area": area, "wall_s": round(wall, 1), "steps": model.receipts}
(out / "proof-px-receipt.json").write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt, indent=1))
