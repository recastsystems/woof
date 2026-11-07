"""Read the WRF v4.6.1 YSU oracle fixture and measure the port against it.

The fixture is written by ``tools/ysu_wrf461_oracle/run_bl_ysu.F90``, which
calls ``bl_ysu_run`` in the byte-unmodified ``phys/physics_mmm/bl_ysu.F90``.
Two CSVs live in ``woof/data/ysu/oracle/``:

``ysu-levels.csv``
    one row per (case, k): every per-level input the scheme was given and every
    per-level word it wrote, plus ``rthblten``, which is
    ``module_bl_ysu.F:452``'s ``ttnp/pi2d`` -- the potential-temperature
    tendency the WRF driver actually hands the solver, and therefore the
    quantity ``kernels/ysu_480.cu``'s ``dtheta`` has to be compared against.
``ysu-surface.csv``
    one row per case: the surface-coupling inputs and ``hpbl``/``kpbl``/
    ``wstar``/``delta``.

Both CSVs also carry the momentum tendencies from a *second* call made with
``ctopo = ctopo2 = 1``.  That second call is not an alternative -- it is what
WRF's own driver does on every column of every default run
(``module_bl_ysu.F:404``), and since development branch it is the arm every
entry point of ``kernels/ysu_480.cu`` takes, so the port's ``du``/``dv`` are
graded against it.  The first, ctopo-absent call (``utnp``/``vtnp``) is
kept as the WRF-against-WRF control that shows the two arms differ.

Nothing in this module knows a tolerance.  It returns measurements; the gate
decides what to do with them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from woof import data_assets

import numpy as np

from woof.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "YSU_ORACLE_DIR",
    "YsuOracleFixture",
    "load_ysu_oracle",
    "ysu_port_outputs",
    "measure_ysu_parity",
]

#: Where the packaged fixture lives.
YSU_ORACLE_DIR = data_assets.verify_oracle_path("ysu/oracle")

#: port output name -> oracle column name, for the per-level fields.
LEVEL_FIELD_MAP = {
    "du": "utnp_ctopo",
    "dv": "vtnp_ctopo",
    "dtheta": "rthblten",
    "dqv": "qvtnp",
    "dqc": "qctnp",
    "dqi": "qitnp",
    "exch_h": "exch_hx",
    "exch_m": "exch_mx",
}

#: port output name -> oracle column name, for the per-column fields.
SURFACE_FIELD_MAP = {
    "hpbl": "hpbl",
    "wstar": "wstar",
    "delta": "delta",
}

#: The ctopo-absent call (bl_ysu.F90:1315, ``ad(1) = 1+fric``), which no WRF
#: run reaches because module_bl_ysu.F:404 always passes ctopo.  The port
#: stopped taking it in development branch; measured against it only to show
#: the port now sits on WRF's default arm and not on this one.
NOCTOPO_FIELD_MAP = {"du": "utnp", "dv": "vtnp"}

#: Loaded but not compared to anything by itself.  ``ttnp`` is the temperature
#: tendency ``bl_ysu_run`` returns, before ``module_bl_ysu.F:452`` divides it
#: back by ``pi2d``; a gate uses it to show that the last ULP between the
#: kernel's ``dtheta`` and ``rthblten`` is that multiply-then-divide and
#: nothing else.
EXTRA_REFERENCE_COLUMNS = ("ttnp",)


@dataclass(frozen=True)
class YsuOracleFixture:
    """One fixture: ``nz`` levels by ``ncase`` columns, laid out for the port."""

    nz: int
    cases: tuple[int, ...]
    topdown: tuple[int, ...]
    dt: float
    inputs: dict[str, np.ndarray]
    level_reference: dict[str, np.ndarray]
    surface_reference: dict[str, np.ndarray]
    kpbl_reference: np.ndarray

    @property
    def ncase(self) -> int:
        return len(self.cases)


def load_ysu_oracle(directory: Path | None = None) -> YsuOracleFixture:
    """Load the two CSVs into the port's ``(nz, 1, ncase)`` layout.

    ``theta`` is not in the fixture because WRF's YSU is not given one: it
    receives temperature and computes ``thx = tx/pi2d`` itself
    (``bl_ysu.F90:419``).  The port is handed ``theta`` by the model, so the
    fixture's ``theta`` is reconstructed here with the *same* float32 divide.
    That makes the port's ``thx`` bit-identical to WRF's by construction, and
    leaves any difference downstream attributable to the port rather than to
    how the comparison was set up.
    """
    directory = Path(directory) if directory is not None else YSU_ORACLE_DIR
    data_assets.require_verify_oracle(directory)
    with (directory / "ysu-levels.csv").open(newline="", encoding="ascii") as fh:
        level_rows = list(csv.DictReader(fh))
    with (directory / "ysu-surface.csv").open(newline="", encoding="ascii") as fh:
        surface_rows = list(csv.DictReader(fh))

    cases = tuple(int(row["case"]) for row in surface_rows)
    nz = int(surface_rows[0]["nz"])
    if any(int(row["nz"]) != nz for row in surface_rows):
        raise ValueError("fixture mixes column depths; the port takes one nz")
    dts = {row["dt"].strip() for row in surface_rows}
    if len(dts) != 1:
        raise ValueError("fixture mixes timesteps; launch_ysu takes one dt")
    dt = float(np.float32(dts.pop()))
    ncase = len(cases)
    if len(level_rows) != nz * ncase:
        raise ValueError(
            f"expected {nz * ncase} level rows, found {len(level_rows)}")

    order = {case: i for i, case in enumerate(cases)}
    grid = np.empty((nz, 1, ncase), dtype=np.float32)

    def level(column: str) -> np.ndarray:
        out = grid.copy()
        for row in level_rows:
            out[int(row["k"]) - 1, 0, order[int(row["case"])]] = np.float32(
                row[column])
        return np.ascontiguousarray(out)

    p_interface = np.empty((nz + 1, 1, ncase), dtype=np.float32)
    for row in level_rows:
        k, col = int(row["k"]) - 1, order[int(row["case"])]
        p_interface[k, 0, col] = np.float32(row["p2di_k"])
        if k == nz - 1:
            p_interface[nz, 0, col] = np.float32(row["p2di_kp1"])

    tx, pi2d = level("tx"), level("pi2d")
    inputs = {
        "u": level("ux"),
        "v": level("vx"),
        # bl_ysu.F90:419 -- thx(i,k) = tx(i,k)/pi2d(i,k)
        "theta": np.ascontiguousarray(tx / pi2d),
        "qv": level("qvx"),
        "qc": level("qcx"),
        "qi": level("qix"),
        "p": level("p2d"),
        "p_interface": np.ascontiguousarray(p_interface),
        "exner": pi2d,
        "dz": level("dz8w"),
        "rthraten": level("rthraten"),
    }
    for name in ("psfcpa", "znt", "ust", "hfx", "qfx", "wspd", "br",
                 "psim", "psih", "xland", "u10", "v10"):
        column = np.empty((1, ncase), dtype=np.float32)
        for row in surface_rows:
            column[0, order[int(row["case"])]] = np.float32(row[name])
        key = "psfc" if name == "psfcpa" else name
        inputs[key] = np.ascontiguousarray(column)

    # Keyed by the oracle's own column name: du/dv are compared against both
    # ``utnp`` and ``utnp_ctopo``, so a port-name key would collide.
    level_reference = {
        column: level(column)
        for column in sorted(set(LEVEL_FIELD_MAP.values())
                             | set(NOCTOPO_FIELD_MAP.values())
                             | set(EXTRA_REFERENCE_COLUMNS))
    }

    surface_reference = {}
    for column in ("hpbl", "wstar", "delta"):
        values = np.empty((1, ncase), dtype=np.float32)
        for row in surface_rows:
            values[0, order[int(row["case"])]] = np.float32(row[column])
        surface_reference[column] = np.ascontiguousarray(values)
    kpbl_reference = np.empty((1, ncase), dtype=np.int32)
    for row in surface_rows:
        kpbl_reference[0, order[int(row["case"])]] = int(row["kpbl"])

    topdown = tuple(int(row["topdown"]) for row in surface_rows)
    return YsuOracleFixture(
        nz=nz, cases=cases, topdown=topdown, dt=dt, inputs=inputs,
        level_reference=level_reference,
        surface_reference=surface_reference,
        kpbl_reference=np.ascontiguousarray(kpbl_reference),
    )


def ysu_port_outputs(fixture: YsuOracleFixture, *, ysu_version="wrf_461") -> dict[str, np.ndarray]:
    """Run ``launch_ysu`` on the fixture and bring the results back to host.

    ``ysu_topdown_pblmix`` is a launch-wide integer while the fixture varies it
    per case, so the kernel is launched once for each value and the columns are
    stitched back together by the flag the oracle recorded.
    """
    import cupy as cp

    from woof.core.ysu import launch_ysu

    device = {name: cp.asarray(value) for name, value in fixture.inputs.items()}
    flags = np.asarray(fixture.topdown, dtype=np.int32)
    merged: dict[str, np.ndarray] = {}
    for flag in sorted(set(int(f) for f in flags)):
        out = launch_ysu(
            device["u"], device["v"], device["theta"], device["qv"],
            device["qc"], device["qi"], device["p"], device["p_interface"],
            device["exner"], device["dz"], device["rthraten"],
            psfc=device["psfc"], znt=device["znt"], ust=device["ust"],
            hfx=device["hfx"], qfx=device["qfx"], wspd=device["wspd"],
            br=device["br"], psim=device["psim"], psih=device["psih"],
            xland=device["xland"], u10=device["u10"], v10=device["v10"],
            dt=fixture.dt, ysu_topdown_pblmix=flag,
            ysu_version=ysu_version,
        )
        host = {name: cp.asnumpy(value) for name, value in out.items()}
        take = flags == flag
        for name, value in host.items():
            if name not in merged:
                merged[name] = np.zeros_like(value)
            merged[name][..., take] = value[..., take]
    return merged


@dataclass(frozen=True)
class FieldParity:
    """The measurement for one output field."""

    field: str
    reference: str
    max_ulp: int
    differing: int
    total: int
    worst_case: int
    worst_level: int | None
    got: float
    want: float

    def __str__(self) -> str:
        where = (f"case {self.worst_case}"
                 + ("" if self.worst_level is None else f" k={self.worst_level}"))
        return (f"{self.field:<8s} vs {self.reference:<12s} "
                f"max_ulp={self.max_ulp:<12d} "
                f"differing={self.differing}/{self.total} "
                f"worst at {where}: got {self.got!r} want {self.want!r}")


def _worst(fixture, field, reference, got, want) -> FieldParity:
    distance = fp32_ulp_distance(got, want)
    flat = int(np.argmax(distance)) if distance.size else 0
    index = np.unravel_index(flat, distance.shape) if distance.size else ()
    if got.ndim == 3:
        level, _, col = index
        worst_level = int(level) + 1
    else:
        _, col = index
        worst_level = None
    return FieldParity(
        field=field, reference=reference,
        max_ulp=int(distance.max()) if distance.size else 0,
        differing=int(np.count_nonzero(
            got.view(np.uint32) != want.view(np.uint32))),
        total=int(got.size),
        worst_case=fixture.cases[int(col)],
        worst_level=worst_level,
        got=float(got.ravel()[flat]), want=float(want.ravel()[flat]),
    )


def measure_ysu_parity(fixture: YsuOracleFixture,
                       port: dict[str, np.ndarray]) -> dict[str, FieldParity]:
    """ULP distance from every port output to the word the WRF module wrote."""
    result: dict[str, FieldParity] = {}
    for name, column in LEVEL_FIELD_MAP.items():
        result[name] = _worst(fixture, name, column,
                              np.ascontiguousarray(port[name], np.float32),
                              fixture.level_reference[column])
    for name, column in SURFACE_FIELD_MAP.items():
        result[name] = _worst(fixture, name, column,
                              np.ascontiguousarray(port[name], np.float32),
                              fixture.surface_reference[column])
    for name, column in NOCTOPO_FIELD_MAP.items():
        result[f"{name}@noctopo"] = _worst(
            fixture, name, column,
            np.ascontiguousarray(port[name], np.float32),
            fixture.level_reference[column])
    return result


def kpbl_disagreements(fixture: YsuOracleFixture,
                       port: dict[str, np.ndarray]) -> dict[int, tuple[int, int]]:
    """Cases where the port's one-based ``kpbl`` is not WRF's, keyed by case."""
    got = np.asarray(port["kpbl"], dtype=np.int32).reshape(-1)
    want = fixture.kpbl_reference.reshape(-1)
    return {fixture.cases[i]: (int(got[i]), int(want[i]))
            for i in range(got.size) if got[i] != want[i]}
