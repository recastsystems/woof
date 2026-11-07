"""Read the WRF v4.6.1 Shin-Hong oracle fixture and measure the port against it.

The fixture is written by ``tools/shinhong_wrf461_oracle/run_bl_shinhong.F90``,
which calls ``shinhong`` -- the 3-D wrapper itself, so the ndiff-packed
moisture array, the ``rthblten = ttnp/pi`` identity
(module_bl_shinhong.F:253) and the ``u10/v10`` ctopo2 blend (:1472) are all
inside the pinned boundary.  Three CSVs live in
``woof/data/shinhong/oracle/``:

``shinhong-levels.csv``
    one row per (case, dxi, k): every per-level input, the arm-A outputs
    (``ctopo = ctopo2 = 1``, the Registry topo_wind=0 default), and the arm-B
    (``ctopo = 0.85, ctopo2 = 0.7``) momentum tendencies and ``exch_h``.
``shinhong-surface.csv``
    one row per (case, dxi): surface-coupling inputs, per-column ``dx``,
    ``dy``, ``dt`` and ``tke_diag`` (all four are fixture axes here -- the
    whole point of Shin-Hong is that the answer moves with dx), and both
    arms' ``hpbl``/``kpbl``/``wstar``/``delta``/``u10``/``v10``.
``shinhong-partition.csv``
    the five partition functions probed directly on a d x h grid spanning
    both clamps and the ``h == 0`` early-out.

The fixture column axis is the flattened (case, dxi) pairs: 30 cases x 6
grid spacings = 180 columns, laid out ``(nz, 1, ncol)`` float32 exactly like
the YSU fixture, in file order.

NaN lanes: gfortran's ES24.16E3 writes NaN as the sign-less token ``NaN``,
so the CSV cannot carry the payload or sign of the 0/0 NaN WRF produced on
case 13 (x86 default QNaN, 0xFFC00000); parsing lands on 0x7FC00000.  The
measurement therefore reports NaN lanes separately -- geometry (which lanes)
and count -- and the ULP maximum is taken over the lanes where neither side
is NaN.  ``woof.core.fp32_ulp`` would report any NaN-against-anything as
:data:`~woof.core.fp32_ulp.MISMATCH`, which is correct for a gate but would
bury the real question here: whether the port's NaN GEOMETRY is WRF's.

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
from woof.verify.shinhong_ref import np_shinhong_column

__all__ = [
    "SHINHONG_ORACLE_DIR",
    "ARM_A",
    "ARM_B",
    "LEVEL_FIELD_MAP",
    "SURFACE_FIELD_MAP",
    "TOPO_LEVEL_FIELD_MAP",
    "TOPO_SURFACE_FIELD_MAP",
    "ShinhongOracleFixture",
    "load_shinhong_oracle",
    "shinhong_ref_outputs",
    "measure_shinhong_parity",
    "kpbl_disagreements",
]

#: Where the packaged fixture lives.
SHINHONG_ORACLE_DIR = data_assets.verify_oracle_path("shinhong/oracle")

#: The two oracle arms: (ctopo, ctopo2).
ARM_A = (np.float32(1.0), np.float32(1.0))
ARM_B = (np.float32(0.85), np.float32(0.7))

#: port output name -> oracle column name, per-level, arm A.
LEVEL_FIELD_MAP = {
    "rublten": "rublten",
    "rvblten": "rvblten",
    "rthblten": "rthblten",
    "rqvblten": "rqvblten",
    "rqcblten": "rqcblten",
    "rqiblten": "rqiblten",
    "exch_h": "exch_h",
    "tke": "tke_out",
    "el": "el_out",
}

#: port output name -> oracle column name, per-column, arm A.
SURFACE_FIELD_MAP = {
    "hpbl": "hpbl",
    "wstar": "wstar",
    "delta": "delta",
    "u10": "u10_out",
    "v10": "v10_out",
}

#: The arm-B fields the oracle recorded (the ctopo algebra touches momentum
#: and the u10/v10 blend; everything else is bitwise arm A's).
TOPO_LEVEL_FIELD_MAP = {
    "rublten": "rublten_tw",
    "rvblten": "rvblten_tw",
    "exch_h": "exch_h_tw",
}
TOPO_SURFACE_FIELD_MAP = {
    "hpbl": "hpbl_tw",
    "wstar": "wstar_tw",
    "delta": "delta_tw",
    "u10": "u10_tw",
    "v10": "v10_tw",
}

_LEVEL_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "p2d", "pi2d",
                 "dz8w", "tke_in")
_SURFACE_INPUTS = ("psfc", "znt", "ust", "hfx", "qfx", "wspd_in", "br",
                   "psim", "psih", "xland", "corf", "u10_in", "v10_in",
                   "dx", "dy", "dt")
_LEVEL_REFERENCE = tuple(LEVEL_FIELD_MAP.values()) + tuple(
    TOPO_LEVEL_FIELD_MAP.values())
_SURFACE_REFERENCE = tuple(SURFACE_FIELD_MAP.values()) + tuple(
    TOPO_SURFACE_FIELD_MAP.values()) + ("wspd_out",)


@dataclass(frozen=True)
class ShinhongOracleFixture:
    """One fixture: ``nz`` levels by ``ncol`` flattened (case, dxi) columns."""

    nz: int
    columns: tuple[tuple[int, int], ...]
    inputs: dict[str, np.ndarray]
    tke_diag: np.ndarray
    level_reference: dict[str, np.ndarray]
    surface_reference: dict[str, np.ndarray]
    kpbl_reference: np.ndarray
    kpbl_tw_reference: np.ndarray
    partition: dict[str, np.ndarray]

    @property
    def ncol(self) -> int:
        return len(self.columns)

    @property
    def case_ids(self) -> tuple[int, ...]:
        return tuple(sorted({case for case, _ in self.columns}))

    def case_mask(self, *cases: int) -> np.ndarray:
        """Boolean mask over the column axis selecting whole cases."""
        wanted = set(cases)
        return np.asarray([case in wanted for case, _ in self.columns])


def load_shinhong_oracle(directory: Path | None = None) -> ShinhongOracleFixture:
    """Load the three CSVs into the ``(nz, 1, ncol)`` layout.

    Unlike the YSU loader nothing is reconstructed here: the fixture calls
    the 3-D wrapper, whose inputs are exactly what the port takes, so every
    array below is a straight float32 parse of the oracle's ES24.16E3 text
    (full float32 round-trip precision).
    """
    directory = Path(directory) if directory is not None else SHINHONG_ORACLE_DIR
    data_assets.require_verify_oracle(directory)
    with (directory / "shinhong-levels.csv").open(
            newline="", encoding="ascii") as fh:
        level_rows = list(csv.DictReader(fh))
    with (directory / "shinhong-surface.csv").open(
            newline="", encoding="ascii") as fh:
        surface_rows = list(csv.DictReader(fh))
    with (directory / "shinhong-partition.csv").open(
            newline="", encoding="ascii") as fh:
        partition_rows = list(csv.DictReader(fh))

    columns = tuple((int(row["case"]), int(row["dxi"])) for row in surface_rows)
    if len(set(columns)) != len(columns):
        raise ValueError("fixture repeats a (case, dxi) column")
    nz = int(surface_rows[0]["nz"])
    if any(int(row["nz"]) != nz for row in surface_rows):
        raise ValueError("fixture mixes column depths; the port takes one nz")
    ncol = len(columns)
    if len(level_rows) != nz * ncol:
        raise ValueError(
            f"expected {nz * ncol} level rows, found {len(level_rows)}")

    order = {key: i for i, key in enumerate(columns)}

    def level(column: str) -> np.ndarray:
        out = np.empty((nz, 1, ncol), dtype=np.float32)
        for row in level_rows:
            j = order[(int(row["case"]), int(row["dxi"]))]
            out[int(row["k"]) - 1, 0, j] = np.float32(row[column])
        return np.ascontiguousarray(out)

    def surface(column: str, dtype=np.float32) -> np.ndarray:
        out = np.empty((1, ncol), dtype=dtype)
        for row in surface_rows:
            j = order[(int(row["case"]), int(row["dxi"]))]
            out[0, j] = dtype(row[column])
        return np.ascontiguousarray(out)

    inputs: dict[str, np.ndarray] = {name: level(name) for name in _LEVEL_INPUTS}
    p_interface = np.empty((nz + 1, 1, ncol), dtype=np.float32)
    for row in level_rows:
        k = int(row["k"]) - 1
        j = order[(int(row["case"]), int(row["dxi"]))]
        p_interface[k, 0, j] = np.float32(row["p2di_k"])
        if k == nz - 1:
            p_interface[nz, 0, j] = np.float32(row["p2di_kp1"])
    inputs["p_interface"] = np.ascontiguousarray(p_interface)
    for name in _SURFACE_INPUTS:
        inputs[name] = surface(name)

    level_reference = {name: level(name) for name in _LEVEL_REFERENCE}
    surface_reference = {name: surface(name) for name in _SURFACE_REFERENCE}

    partition = {
        name: np.asarray([np.float32(row[name]) for row in partition_rows],
                         dtype=np.float32)
        for name in ("d", "h", "pu", "pq", "pthnl", "pthl", "ptke")
    }

    return ShinhongOracleFixture(
        nz=nz,
        columns=columns,
        inputs=inputs,
        tke_diag=surface("tke_diag", dtype=np.int32),
        level_reference=level_reference,
        surface_reference=surface_reference,
        kpbl_reference=surface("kpbl", dtype=np.int32),
        kpbl_tw_reference=surface("kpbl_tw", dtype=np.int32),
        partition=partition,
    )


_LEVEL_OUTPUTS = tuple(LEVEL_FIELD_MAP)
_SURFACE_OUTPUTS = tuple(SURFACE_FIELD_MAP)


def shinhong_ref_outputs(fixture: ShinhongOracleFixture,
                         arm=ARM_A) -> dict[str, np.ndarray]:
    """Run ``np_shinhong_column`` on every fixture column for one arm.

    ``arm`` is (ctopo, ctopo2): :data:`ARM_A` is WRF's Registry topo_wind=0
    default, :data:`ARM_B` the oracle's (0.85, 0.7) probe.  dx, dy, dt and
    tke_diag come from the fixture per column -- they are axes, not
    constants.
    """
    nz, ncol = fixture.nz, fixture.ncol
    out = {name: np.zeros((nz, 1, ncol), dtype=np.float32)
           for name in _LEVEL_OUTPUTS}
    out.update({name: np.zeros((1, ncol), dtype=np.float32)
                for name in _SURFACE_OUTPUTS})
    out["kpbl"] = np.zeros((1, ncol), dtype=np.int32)
    x = fixture.inputs
    for j in range(ncol):
        column = np_shinhong_column(
            x["u"][:, 0, j], x["v"][:, 0, j], x["t"][:, 0, j],
            x["qv"][:, 0, j], x["qc"][:, 0, j], x["qi"][:, 0, j],
            x["p2d"][:, 0, j], x["pi2d"][:, 0, j],
            x["p_interface"][:, 0, j], x["dz8w"][:, 0, j],
            x["tke_in"][:, 0, j],
            psfc=x["psfc"][0, j], znt=x["znt"][0, j], ust=x["ust"][0, j],
            hfx=x["hfx"][0, j], qfx=x["qfx"][0, j], wspd=x["wspd_in"][0, j],
            br=x["br"][0, j], psim=x["psim"][0, j], psih=x["psih"][0, j],
            xland=x["xland"][0, j], corf=x["corf"][0, j],
            u10=x["u10_in"][0, j], v10=x["v10_in"][0, j],
            dt=x["dt"][0, j], dx=x["dx"][0, j], dy=x["dy"][0, j],
            tke_diag=int(fixture.tke_diag[0, j]),
            ctopo=arm[0], ctopo2=arm[1])
        for name in _LEVEL_OUTPUTS:
            out[name][:, 0, j] = column[name]
        for name in _SURFACE_OUTPUTS:
            out[name][0, j] = column[name]
        out["kpbl"][0, j] = column["kpbl"]
    return out


@dataclass(frozen=True)
class FieldParity:
    """The measurement for one output field.

    ``max_ulp`` is over the lanes where neither side is NaN; the NaN lanes
    are reported as geometry (``nan_mask_matches``) and counts, because the
    fixture text cannot carry NaN bits (see the module docstring).
    """

    field: str
    reference: str
    max_ulp: int
    differing: int
    total: int
    nan_reference: int
    nan_port: int
    nan_mask_matches: bool
    worst_case: int
    worst_dxi: int
    worst_level: int | None
    got: float
    want: float

    def __str__(self) -> str:
        where = (f"case {self.worst_case} dxi {self.worst_dxi}"
                 + ("" if self.worst_level is None
                    else f" k={self.worst_level}"))
        nan = ""
        if self.nan_reference or self.nan_port:
            agree = "same lanes" if self.nan_mask_matches else "LANES DIFFER"
            nan = (f" nan_ref={self.nan_reference}"
                   f" nan_port={self.nan_port} ({agree})")
        return (f"{self.field:<9s} vs {self.reference:<12s} "
                f"max_ulp={self.max_ulp:<12d} "
                f"differing={self.differing}/{self.total}{nan} "
                f"worst at {where}: got {self.got!r} want {self.want!r}")


def _worst(fixture: ShinhongOracleFixture, field: str, reference: str,
           got: np.ndarray, want: np.ndarray) -> FieldParity:
    got = np.ascontiguousarray(got, np.float32)
    want = np.ascontiguousarray(want, np.float32)
    nan_got = np.isnan(got)
    nan_want = np.isnan(want)
    arithmetic = ~(nan_got | nan_want)
    distance = np.where(arithmetic, fp32_ulp_distance(got, want), np.int64(-1))
    flat = int(np.argmax(distance)) if distance.size else 0
    index = np.unravel_index(flat, distance.shape) if distance.size else ()
    if got.ndim == 3:
        level, _, col = index
        worst_level = int(level) + 1
    else:
        _, col = index
        worst_level = None
    case, dxi = fixture.columns[int(col)]
    return FieldParity(
        field=field, reference=reference,
        max_ulp=int(max(distance.max(), 0)) if distance.size else 0,
        differing=int(np.count_nonzero(
            (got.view(np.uint32) != want.view(np.uint32)) & arithmetic)),
        total=int(got.size),
        nan_reference=int(nan_want.sum()),
        nan_port=int(nan_got.sum()),
        nan_mask_matches=bool(np.array_equal(nan_got, nan_want)),
        worst_case=case, worst_dxi=dxi, worst_level=worst_level,
        got=float(got.ravel()[flat]), want=float(want.ravel()[flat]),
    )


def measure_shinhong_parity(fixture: ShinhongOracleFixture,
                            port: dict[str, np.ndarray],
                            arm=ARM_A) -> dict[str, FieldParity]:
    """ULP distance from every port output to the word the WRF module wrote."""
    level_map = LEVEL_FIELD_MAP if arm == ARM_A else TOPO_LEVEL_FIELD_MAP
    surface_map = SURFACE_FIELD_MAP if arm == ARM_A else TOPO_SURFACE_FIELD_MAP
    result: dict[str, FieldParity] = {}
    for name, column in level_map.items():
        result[name] = _worst(fixture, name, column, port[name],
                              fixture.level_reference[column])
    for name, column in surface_map.items():
        result[name] = _worst(fixture, name, column, port[name],
                              fixture.surface_reference[column])
    return result


def kpbl_disagreements(fixture: ShinhongOracleFixture,
                       port: dict[str, np.ndarray],
                       arm=ARM_A) -> dict[tuple[int, int], tuple[int, int]]:
    """Columns where the port's one-based kpbl is not WRF's, keyed (case, dxi)."""
    reference = (fixture.kpbl_reference if arm == ARM_A
                 else fixture.kpbl_tw_reference)
    got = np.asarray(port["kpbl"], dtype=np.int32).reshape(-1)
    want = reference.reshape(-1)
    return {fixture.columns[i]: (int(got[i]), int(want[i]))
            for i in range(got.size) if got[i] != want[i]}
