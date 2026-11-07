"""CAM column ABI and allocation shapes, importable without CUDA."""
from functools import lru_cache
import json
from pathlib import Path


DEFAULT_WORKERS = 8192
BLOCK_THREADS = 32
CAM_FIELDS_2D = ('swcf','lwcf','coszr')
CAM_FIELDS_3D = ('cemiss','taucldc','taucldi','rthratenlwc','rthratenswc')
RETAINED_FIELDS = frozenset({"abstot_3d", "absnxt_3d", "emstot_3d"})

# Actual cam_driver local-memory ceilings over fast and strict compilation.
# 2026-11-01, sm_120, NVRTC 13.4.59 (CL-38657139). Source-default 32
# levels: 1920/2048 B; specialized 34 and 49: 1904/2064 B. Receipt command:
# probe_batch005_cam_levels.py; logs/batch005-cam-levels-{fast,strict}.json
# in the port-integrate evidence. Other depths have no measurement here.
MEASURED_LOCAL_FRAME_BYTES = {32: 2048, 34: 2064, 49: 2064}


@lru_cache(maxsize=1)
def layout():
    root = Path(__file__).parent
    column = json.loads((root / "kernels/cam3_emissivity.json").read_text())["camrad"]
    state = json.loads((root.parent / "data/cam3/state-layout.json").read_text())
    return column, max(field["offset"] + field["bytes"] for field in state)


def expression_count(expression, nz):
    return eval(expression, {"__builtins__": {}}, {"CAM_LEVELS": int(nz)})


@lru_cache(maxsize=None)
def column_parameters(nz):
    """Immutable ABI names, scalar types and widths for one model depth."""
    return tuple((field['name'], field['dtype'], expression_count(
        field.get('count_expression', str(field['count'])), nz))
        for field in layout()[0]['parameters'])


def local_frame_bound(nz):
    """Conservative sum of automatic arrays and scalar/ABI overhead.

    The emitter records this expression beside the workspace contract. It
    includes mutually exclusive branches within a frame and every device
    call path, including the independent absorption rows.
    """
    return 16384 + max(expression_count(value,nz) for value in
                       layout()[0]['local_frame_bound_expressions'])


def worker_count(ncol, limit=DEFAULT_WORKERS):
    """Warp-rounded workers that spread the columns over equal rounds.

    Each worker walks columns serially, so the call costs the number of
    rounds a worker makes.  Spreading those rounds evenly keeps that count
    at its minimum for ``limit`` and allocates no worker whose last round
    would idle (50,451 columns: 7,232 workers, not 8,192, same 7 rounds).
    """
    if int(ncol) < 1 or int(limit) < 1:
        raise ValueError("CAM needs a positive column count and worker limit")
    ncol = int(ncol)
    cap = min(ncol, int(limit))
    rounds = -(-ncol // cap)
    return ((-(-ncol // rounds) + BLOCK_THREADS - 1) // BLOCK_THREADS) * BLOCK_THREADS


def allocation_shapes(nz, ny, nx, limit=DEFAULT_WORKERS):
    """Resident storage and a conservative bound for simultaneous packing.

    Kernel arguments, prior retained arrays, and packed inputs coexist.
    Preflight accounts for device frames separately. Its compiler recording
    covers only the depths in MEASURED_LOCAL_FRAME_BYTES; other CAM_LEVELS
    specializations are priced at the explicit assumed frame bound.
    """
    meta, state_bytes = layout()
    ncol = int(ny) * int(nx)
    rows = {
        "cam3/state": ("physics", (state_bytes,), 1),
        "cam3/ozone": ("physics", (ncol, 59*13), 4),
        "cam3/workspace": ("transient", (worker_count(ncol, limit),
            expression_count(meta["workspace_expression"], nz)), 1),
        "cam3/absorption_workspace": ("transient", (worker_count(ncol, limit)*(int(nz)+1),
            expression_count(meta["absorption_workspace_expression"], nz)), 1),
        "cam3/absorption_operands": ("transient", (worker_count(ncol, limit),41), 8),
        "cam3/absorption_results": ("transient", (worker_count(ncol, limit),
            (int(nz)+1)**2+int(nz)*4+int(nz)+1), 8),
    }
    widths = {"float": 4, "double": 8, "int": 4}
    for field in meta["parameters"]:
        shape = (ncol, expression_count(field.get("count_expression", str(field["count"])), nz))
        rows["cam3/argument/" + field["name"]] = ("transient", shape, widths[field["dtype"]])
        if field["name"] in RETAINED_FIELDS:
            rows["cam3/retained/" + field["name"]] = ("physics", shape, 4)
    rows.update({
        "cam3/mass_packing": ("transient", (24, ncol, nz), 4),
        "cam3/interface_packing": ("transient", (ncol, nz+1), 4),
        "cam3/surface_packing": ("transient", (12, ncol), 4),
        "cam3/aerosol_packing": ("physics", (2, ncol, 29*13), 4),
        "cam3/aerosol_pressure": ("physics", (ncol,), 4),
        "cam3/table_packing": ("physics", (ncol, 59+29), 4),
        "cam3/gas_packing": ("transient", (ncol, 5), 8),
    })
    return rows
