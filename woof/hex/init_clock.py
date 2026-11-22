"""Read the valid clock of a single native initialization state."""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np


def _parse(value):
    value = str(value).strip().replace("_", "T").replace("Z", "+00:00")
    when = datetime.fromisoformat(value)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def valid_time(path):
    """Prefer the state's one populated xtime; otherwise use config_start_time.

    A blank template clock is allowed. Multiple populated times are ambiguous
    for an initialization state, even when the file's start attribute exists.
    """
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if value.get("schema") != "gpuwm-hex.mapped-init/v1":
            raise ValueError("unrecognized native initialization manifest")
        return _parse(value["valid"])
    from netCDF4 import Dataset
    with Dataset(path) as state:
        state.set_auto_maskandscale(False)
        if "xtime" in state.variables:
            words = np.asarray(state["xtime"][:])
            if words.ndim not in (1, 2) or words.dtype.kind != "S":
                raise ValueError("native xtime must be a character row or rows")
            rows = words.reshape(1, -1) if words.ndim == 1 else words
            clocks = [row.tobytes().decode("ascii").strip(" \x00") for row in rows]
            clocks = [clock for clock in clocks if clock]
            if len(clocks) > 1:
                raise ValueError("native initialization carries multiple clocks")
            if clocks:
                return _parse(clocks[0])
        if "config_start_time" not in state.ncattrs():
            raise ValueError("native initialization has no valid clock")
        return _parse(state.getncattr("config_start_time"))
