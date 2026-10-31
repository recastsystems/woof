"""Does the VRAM inventory name every array the physics driver allocates?

``preflight.physics_array_shapes`` is the physics half of every resident,
tile and lifecycle estimate, and it is a hand-kept list beside the
allocations it prices.  With the MYJ pair selected (``bl_pbl_physics = 2``,
``sf_sfclay_physics = 2``) it named neither TKE_MYJ/EL_MYJ nor 17 of the
Eta layer's surface planes, while ``initialize_physics`` allocated all 19:
4*C*(2*nz + 17) = 932,184,064 bytes (0.868 GiB) at 1792x1024x55 that no
estimate saw, so a run admitted with less spare than that allocated past
its estimate.

The census closes the class, not the instance.  For every surface layer,
PBL and land-surface selector the config door admits, at both PBL
cadences, ``tests/physics_allocation_census.py`` builds the real driver on
a counting NumPy stand-in for cupy in a child interpreter (no card, no
cupy needed) and reports every buffer the construction left on it.  Each
one must be in the inventory under the same name, with the same shape and
four bytes a value.  The next field a scheme adds without pricing fails
here instead of in a run.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from physics_allocation_census import admitted_cells, cell_config, cell_id

REPO = Path(__file__).resolve().parents[1]
_CHILD = Path(__file__).resolve().with_name("physics_allocation_census.py")
_CELLS = admitted_cells()


@pytest.fixture(scope="module")
def census() -> dict[str, dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # The child never imports the real cupy; hiding every device as well
    # means nothing it reaches can open one either.
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    done = subprocess.run(
        [sys.executable, str(_CHILD)], capture_output=True, text=True,
        env=env, cwd=str(REPO), timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout)


def test_the_census_sweeps_every_selector_the_door_admits():
    """The sweep is the whole selector space, not a sample of it.

    Every surface layer (bar "off", which the door admits only with no
    surface physics at all), every PBL scheme and every land-surface
    scheme the config door names must reach at least one built driver, so
    a selector that gains a driver field is census-covered the day it
    exists.
    """
    from woof.config import (LAND_SURFACE_SCHEMES, PBL_SCHEMES,
                              SURFACE_LAYER_SCHEMES)

    assert set(SURFACE_LAYER_SCHEMES) - {0} <= {c[0] for c in _CELLS}
    assert set(PBL_SCHEMES) <= {c[1] for c in _CELLS}
    assert set(LAND_SURFACE_SCHEMES) <= {c[2] for c in _CELLS}
    assert (2, 2, 2, 0.0) in _CELLS  # the MYJ pair itself


@pytest.mark.parametrize("cell", _CELLS, ids=[cell_id(c) for c in _CELLS])
def test_every_driver_array_is_in_the_vram_inventory(census, cell):
    row = census[cell_id(cell)]
    assert "error" not in row, row.get("error")
    inventory = {key: tuple(shape) for key, shape in row["inventory"].items()}
    allocated = {path: (tuple(shape), nbytes)
                 for path, (shape, nbytes) in row["allocated"].items()}

    unpriced = sorted(path for path in allocated if path not in inventory)
    assert not unpriced, (
        f"{cell_id(cell)}: initialize_physics allocates {unpriced} and "
        "preflight.physics_array_shapes does not price them")
    wrong = {path: (shape, inventory[path])
             for path, (shape, _) in allocated.items()
             if inventory[path] != shape}
    assert not wrong, f"allocated vs priced shapes differ: {wrong}"
    wide = {path: nbytes for path, (shape, nbytes) in allocated.items()
            if nbytes != 4 * math.prod(shape)}
    assert not wide, f"the inventory prices four bytes a value: {wide}"

    # The one inventory group construction does not make: at a positive
    # PBL cadence YSU (pbl 1, the only scheme whose driver path creates
    # PhysicsDriver.last_ysu) retains its raw output from the first PBL
    # call on, so it is priced but not yet allocated.  Every other PBL
    # leaves last_ysu None, so a priced last_ysu there is a phantom.
    from woof.core.physics_inventory import physics_retains_ysu_output

    config = cell_config(cell)
    later = ("last_ysu/" if config.bl_pbl_physics == 1
             and config.bldt > 0.0 else None)
    assert physics_retains_ysu_output(config) == (later is not None)
    phantom = sorted(key for key in inventory if key not in allocated
                     and not (later and key.startswith(later)))
    assert not phantom, (
        f"{cell_id(cell)}: priced but never allocated: {phantom}")


@pytest.mark.parametrize(
    "cell", [c for c in _CELLS if c[3] > 0.0],
    ids=[cell_id(c) for c in _CELLS if c[3] > 0.0])
def test_only_ysu_prices_the_retained_ysu_output(cell):
    """At a positive PBL cadence only YSU is charged for ``last_ysu``.

    ``PhysicsDriver._run_ysu`` is the only path that sets ``last_ysu``;
    MYJ, MYNN, Shin-Hong and SASE leave it None.  Pricing the retained
    set for them over-counts about ten arrays, enough to refuse a run
    that fits at the margin.
    """
    from woof.core import preflight as pf

    config = cell_config(cell)
    shapes = pf.physics_array_shapes(config)
    priced = sorted(key for key in shapes if key.startswith("last_ysu/"))
    if config.bl_pbl_physics == 1:
        assert priced, f"{cell_id(cell)}: YSU's retained output is unpriced"
    else:
        assert not priced, (
            f"{cell_id(cell)}: prices YSU's retained output {priced} for a "
            "PBL scheme that never creates it")


def test_the_myj_pair_prices_its_carried_columns_and_surface_planes():
    """The reviewer's case at the reviewer's shape, without the child.

    The MYJ pair and the YSU/revised-MM5 pair allocate the same driver
    set apart from MYJ's two carried columns and the Eta layer's 16
    surface planes, so the difference between their inventories is exactly
    their 924,844,032 bytes (17 planes until Z0BASE became the shared Z0
    WRF binds, 2026-10-29).
    """
    import dataclasses

    from woof.config import RunConfig
    from woof.core import preflight as pf

    ysu = RunConfig(nx=1792, ny=1024, nz=55, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=15.0, run_seconds=3600.0, moist=True,
                    mp_physics=8, sf_sfclay_physics=1, bl_pbl_physics=1,
                    sf_surface_physics=2)
    myj = dataclasses.replace(ysu, sf_sfclay_physics=2, bl_pbl_physics=2)
    ysu_shapes = pf.physics_array_shapes(ysu)
    myj_shapes = pf.physics_array_shapes(myj)

    added = set(myj_shapes) - set(ysu_shapes)
    assert added == {f"fields/{name}" for name in (
        "tke_myj", "el_myj", "akhs", "akms", "ch", "ct", "mixht", "pshltr",
        "q10", "qshltr", "qz0", "th10", "thz0", "tshltr", "u10e", "uz0",
        "v10e", "vz0")}
    assert not set(ysu_shapes) - set(myj_shapes)
    for name in ("tke_myj", "el_myj"):
        assert myj_shapes[f"fields/{name}"] == (55, 1024, 1792)

    def nbytes(shapes):
        return sum(4 * math.prod(shape) for shape in shapes.values())

    columns = 1792 * 1024
    assert nbytes(myj_shapes) - nbytes(ysu_shapes) == (
        4 * columns * (2 * 55 + 16)) == 924_844_032
