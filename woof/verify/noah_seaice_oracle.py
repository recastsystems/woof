"""Read the WRF v4.6.1 Noah sea-ice oracle fixture and run the port on it.

The fixture is written by ``tools/noah_seaice_wrf461_oracle/run_seaice.F90``,
which calls ``seaice_noah`` in the byte-unmodified
``phys/module_sf_noah_seaice_drv.F`` -- the subroutine
``woof/core/kernels/noah.cu``'s ``noah_seaice_column`` transcribes, prep and
post-update included.  Every column is either a word the harness placed in an
input array before the call or a word ``seaice_noah`` wrote during it;
``sfcprs`` is the one derived echo (WRF forms it from ``p8w3d``), recomputed
by the harness in the same float32 arithmetic.

Nothing here knows a tolerance; ``tests/test_noah_seaice_wrf461_parity.py``
is the gate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from woof import data_assets

import numpy as np

SEAICE_ORACLE_DIR = data_assets.verify_oracle_path("noah/seaice_oracle")

#: fixture file -> the two arguments ``run_seaice`` was invoked with.
SEAICE_ORACLE_FILES = {
    "noah-seaice.csv": dict(frpcpn=False, albedo_default=0.65),
    "noah-seaice-frpcpn.csv": dict(frpcpn=True, albedo_default=0.65),
    "noah-seaice-alb075.csv": dict(frpcpn=False, albedo_default=0.75),
}

#: ``run_seaice.F90``'s fixed arguments.
SEAICE_FIXTURE_DT = 60.0
SEAICE_FIXTURE_XICE_THRESHOLD = 0.5

#: Fields the driver only reads.
SEAICE_FORCING = ("psfc", "sfcprs", "sfctmp", "qv1", "qgh", "glw", "swdown",
                  "rainbl", "sr", "xice", "rib", "chs")
#: Fields the driver may update; the CSV carries ``<name>_in`` and ``<name>``.
SEAICE_STATE = ("cqs2", "chs2", "tsk", "emiss", "albedo", "z0", "snow",
                "snowc", "snowh", "znt", "lh", "hfx", "qfx", "potevp",
                "grdflx", "qsfc", "acsnow", "acsnom", "snopcx", "sfcrunoff",
                "noahres")


@dataclass(frozen=True)
class SeaiceOracleFixture:
    name: str
    switches: dict
    cases: tuple[int, ...]
    inputs: dict[str, np.ndarray]       # (1, ncase); tslb (4, 1, ncase)
    reference: dict[str, np.ndarray]    # same layout


def _row(rows, key) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(
        [np.float32(r[key]) for r in rows], np.float32).reshape(1, -1))


def load_seaice_oracle(name: str = "noah-seaice.csv",
                       directory: Path | None = None) -> SeaiceOracleFixture:
    if name not in SEAICE_ORACLE_FILES:
        raise ValueError(f"{name!r} is not a packaged sea-ice oracle fixture; "
                         f"expected one of {sorted(SEAICE_ORACLE_FILES)}")
    directory = Path(directory) if directory is not None else SEAICE_ORACLE_DIR
    data_assets.require_verify_oracle(directory)
    with (directory / name).open(newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{name} carries no fixture rows")
    inputs = {k: _row(rows, k) for k in SEAICE_FORCING}
    inputs.update({k: _row(rows, f"{k}_in") for k in SEAICE_STATE})
    reference = {k: _row(rows, k) for k in SEAICE_STATE}
    for prefix, target in (("tslb", inputs), ("tslb", reference)):
        suffix = "_in" if target is inputs else ""
        target["tslb"] = np.ascontiguousarray(np.stack(
            [_row(rows, f"{prefix}{k + 1}{suffix}") for k in range(4)]))
    return SeaiceOracleFixture(
        name=name, switches=dict(SEAICE_ORACLE_FILES[name]),
        cases=tuple(int(r["case"]) for r in rows),
        inputs=inputs, reference=reference)


def seaice_port_outputs(fixture: SeaiceOracleFixture) -> dict[str, np.ndarray]:
    """Run ``launch_noah_seaice`` on the fixture's inputs, back to host."""
    import cupy as cp

    from woof.core.noah import launch_noah_seaice

    device = {k: cp.asarray(v.copy()) for k, v in fixture.inputs.items()}
    launch_noah_seaice(device, SEAICE_FIXTURE_DT,
                       frpcpn=fixture.switches["frpcpn"],
                       xice_threshold=SEAICE_FIXTURE_XICE_THRESHOLD,
                       albedo_default=fixture.switches["albedo_default"],
                       check_snow_density=True)
    return {k: cp.asnumpy(device[k]) for k in (*SEAICE_STATE, "tslb")}
