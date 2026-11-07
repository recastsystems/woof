"""Read the WRF v4.6.1 Noah LSM oracle fixture and measure the port against it.

The fixture is written by ``tools/noah_wrf461_oracle/run_lsm.F90``, which calls
``lsm`` in the byte-unmodified ``phys/module_sf_noahdrv.F`` -- the same
subroutine ``woof/core/kernels/noah.cu``'s ``noah_column`` transcribes, prep
and post-update included.  Four CSVs live in ``woof/data/noah/oracle/``, one
per combination of the driver switches ``launch_noah`` exposes.

Every column of every row is either a word the harness placed in an input
array before the call or a word ``lsm`` placed in an output array during it.
The two exceptions are ``sfcprs`` and ``zlvl``, which WRF derives from
``p8w3d`` and ``dz8w`` and woof takes as inputs; the harness recomputes them
in the same float32 arithmetic and writes them out, so they are inputs here
too.

Nothing in this module knows a tolerance.  It returns measurements;
``tests/test_noah_wrf461_parity.py`` decides what to do with them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from woof import data_assets

import numpy as np

from woof.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "NOAH_ORACLE_DIR",
    "NOAH_ORACLE_FILES",
    "NOAH_GLACIAL_ORACLE_FILES",
    "NoahOracleFixture",
    "load_noah_oracle",
    "noah_port_outputs",
    "noah_sfcdiags_outputs",
    "SFCDIAGS_COLUMNS",
    "measure_noah_parity",
]

#: Where the packaged fixture lives.
NOAH_ORACLE_DIR = data_assets.verify_oracle_path("noah/oracle")

#: fixture file -> the driver switches ``run_lsm.F90`` was invoked with.
#: These are the four knobs :func:`woof.core.noah.launch_noah` also takes;
#: ``build.sh`` runs the harness once per row, each in its own process so no
#: pass can inherit the previous pass's mutated state.
NOAH_ORACLE_FILES = {
    "noah-lsm.csv": dict(opt_thcnd=1, frpcpn=False,
                         usemonalb=False, rdlai2d=False),
    "noah-lsm-thcnd2.csv": dict(opt_thcnd=2, frpcpn=False,
                                usemonalb=False, rdlai2d=False),
    "noah-lsm-frpcpn.csv": dict(opt_thcnd=1, frpcpn=True,
                                usemonalb=False, rdlai2d=False),
    "noah-lsm-monalb.csv": dict(opt_thcnd=1, frpcpn=False,
                                usemonalb=True, rdlai2d=True),
}

#: The land-ice fixtures (``run_lsm_glacial.F90``): 24 columns, every one
#: ``ivgtyp == isice``, so WRF's driver takes its ICE == -1 arm and calls
#: SFLX_GLACIAL on all of them.  frpcpn is the one driver switch that reaches
#: SFLX_GLACIAL (through FFROZP); the others do not, so two passes.  These
#: files predate the CHS/CHS2/CQS2 echo and the SFCDIAGS columns of
#: ``run_lsm.F90``; the loader reads the columns a fixture carries.
NOAH_GLACIAL_ORACLE_FILES = {
    "noah-lsm-glacial.csv": dict(opt_thcnd=1, frpcpn=False,
                                 usemonalb=False, rdlai2d=False),
    "noah-lsm-glacial-frpcpn.csv": dict(opt_thcnd=1, frpcpn=True,
                                        usemonalb=False, rdlai2d=False),
}

#: ``run_lsm.F90:42-44``.  The harness fixes these rather than sweeping them,
#: so they are stated here with their source rather than read from the CSV.
NOAH_UA_ORACLE_FILES={}
for _name in ("noah-lsm-ua.csv","noah-lsm-ua-monalb.csv"):
    NOAH_UA_ORACLE_FILES[_name]=dict(opt_thcnd=1,frpcpn=False,
        usemonalb="monalb" in _name,rdlai2d="monalb" in _name,ua_phys=True)

NOAH_UA_ORACLE_FILES["noah-lsm-extended.csv"]=dict(NOAH_ORACLE_FILES["noah-lsm.csv"])
NOAH_UA_ORACLE_FILES["noah-mosaic-ua.csv"]=dict(NOAH_UA_ORACLE_FILES["noah-lsm-ua.csv"],mosaic=True)
NOAH_UA_ORACLE_FILES["noah-mosaic-false.csv"]=dict(NOAH_ORACLE_FILES["noah-lsm.csv"],mosaic=True)
for _name in ("noah-lsm-ua.csv","noah-lsm-ua-monalb.csv","noah-mosaic-ua.csv"):
    NOAH_UA_ORACLE_FILES[_name.replace(".csv","-pc-control.csv")]=dict(NOAH_UA_ORACLE_FILES[_name])
for _name in ("noah-mosaic-ua-two.csv","noah-mosaic-ua-two-pc-control.csv"):
    NOAH_UA_ORACLE_FILES[_name]=dict(NOAH_UA_ORACLE_FILES["noah-mosaic-ua.csv"],mosaic_cat=2)
FIXTURE_DT = 60.0
FIXTURE_ITIMESTEP = 2
#: ``init_soil_depth_2``'s dzs (``share/module_soil_pre.F:1138``), which is
#: also :data:`woof.core.noah.SOIL_LAYER_THICKNESS_M`.
FIXTURE_DZS = (0.10, 0.30, 0.60, 1.00)
#: ``lsm`` is called with these, matching ``launch_noah``'s defaults.
FIXTURE_ISURBAN = 13
FIXTURE_ISICE = 15
FIXTURE_XICE_THRESHOLD = 0.5

#: launch_noah device-field name -> the fixture column holding its input value.
INPUT_COLUMNS = {
    "psfc": "psfc", "sfcprs": "sfcprs", "sfctmp": "sfctmp", "qv1": "qv1",
    "qgh": "qgh", "dz8w1": "dz8w1", "glw": "glw", "swdown": "swdown",
    "rainbl": "rainbl", "sr": "sr", "chs": "chs_in", "cqs2": "cqs2_in",
    "chs2": "chs2_in", "rib": "rib_in", "vegfra": "vegfra",
    "shdmin": "shdmin", "shdmax": "shdmax", "tmn": "tmn", "xland": "xland",
    "xice": "xice", "snoalb": "snoalb", "embck": "embck",
    "tsk": "tsk_in", "hfx": "hfx_in", "qfx": "qfx_in", "lh": "lh_in",
    "grdflx": "grdflx_in", "qsfc": "qsfc_in", "canwat": "canwat_in",
    "snow": "snow_in", "snowc": "snowc_in", "snowh": "snowh_in",
    "albedo": "albedo_in", "albbck": "albbck_in", "emiss": "emiss_in",
    "znt": "znt_in", "z0": "z0_in", "snotime": "snotime_in", "lai": "lai_in",
    "smstav": "smstav_in", "smstot": "smstot_in",
    "sfcrunoff": "sfcrunoff_in", "udrunoff": "udrunoff_in",
    "acsnow": "acsnow_in", "acsnom": "acsnom_in", "snopcx": "snopcx_in",
    "potevp": "potevp_in",
}

#: launch_noah device-field name -> the fixture column holding WRF's answer.
#: ``chs2`` is written by the driver too (``module_sf_noahdrv.F:1275`` sets it
#: to ``cqs2``) but the harness does not echo it back, so it is not measured
#: here; the kernel's copy of that line is checked by tests/test_noah.py.
OUTPUT_COLUMNS = {
    "tsk": "tsk", "hfx": "hfx", "qfx": "qfx", "lh": "lh",
    "grdflx": "grdflx", "qsfc": "qsfc", "canwat": "canwat", "snow": "snow",
    "snowc": "snowc", "snowh": "snowh", "albedo": "albedo",
    "albbck": "albbck", "emiss": "emiss", "znt": "znt", "z0": "z0",
    "snotime": "snotime", "lai": "lai", "smstav": "smstav",
    "smstot": "smstot", "sfcrunoff": "sfcrunoff", "udrunoff": "udrunoff",
    "acsnom": "acsnom", "acsnow": "acsnow", "snopcx": "snopcx",
    "potevp": "potevp", "noahres": "noahres", "chklowq": "chklowq",
    # INOUT arguments of lsm.  On a land column with sf_urban_physics = 0
    # lsm writes CHS2 = CQS2 (module_sf_noahdrv.F:1275) and leaves CHS and
    # CQS2 as they came; the port writes the same three words in place.
    "chs": "chs_out", "chs2": "chs2_out", "cqs2": "cqs2_out",
}

#: SFCDIAGS (phys/module_sf_sfcdiags.F), which module_surface_driver.F:2994
#: calls right after lsm, run by ``run_lsm.F90`` on lsm's own outputs.  The
#: port's counterpart is :mod:`woof.core.noah_sfcdiags`, run on the port's
#: outputs by :func:`noah_sfcdiags_outputs`.
SFCDIAGS_COLUMNS = {"t2": "t2", "th2": "th2", "q2": "q2"}

#: Soil profiles, compared layer by layer.
SOIL_OUTPUT_COLUMNS = ("smois", "tslb", "sh2o", "smcrel")

#: The land-ice columns.  ``ivgtyp == isice`` selects WRF's ``SFLX_GLACIAL``
#: (``module_sf_noahlsm_glacial_only.F``), which ``noah_column`` runs in
#: ``noah_glacial_column`` (it returned without touching the column before
#: 2.9.0).  tests/test_noah_wrf461_parity.py grades them as a population of
#: their own, bitwise.
GLACIAL_CASE_PREDICATE = "ivgtyp == isice"


@dataclass(frozen=True)
class NoahOracleFixture:
    """One fixture file: ``ncase`` independent land columns."""

    name: str
    switches: dict
    cases: tuple[int, ...]
    ivgtyp: np.ndarray
    isltyp: np.ndarray
    inputs: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    soil_reference: dict[str, np.ndarray]
    sfcdiags_reference: dict[str, np.ndarray]

    @property
    def ncase(self) -> int:
        return len(self.cases)

    @property
    def glacial(self) -> np.ndarray:
        """Boolean mask of the land-ice columns (lsm's SFLX_GLACIAL arm)."""
        return self.ivgtyp.reshape(-1) == FIXTURE_ISICE


def _column(rows, key) -> np.ndarray:
    """One fixture column as a ``(1, ncase)`` float32 row of land columns."""
    return np.ascontiguousarray(
        np.asarray([np.float32(row[key]) for row in rows],
                   dtype=np.float32).reshape(1, -1))


def load_noah_oracle(name: str = "noah-lsm.csv",
                     directory: Path | None = None) -> NoahOracleFixture:
    """Load one fixture CSV into ``launch_noah``'s ``(1, ncase)`` layout."""
    known = {**NOAH_ORACLE_FILES, **NOAH_GLACIAL_ORACLE_FILES, **NOAH_UA_ORACLE_FILES}
    if name not in known:
        raise ValueError(f"{name!r} is not a packaged Noah oracle fixture; "
                         f"expected one of {sorted(known)}")
    directory = Path(directory) if directory is not None else NOAH_ORACLE_DIR
    data_assets.require_verify_oracle(directory)
    with (directory / name).open(newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{name} carries no fixture rows")

    cases = tuple(int(row["case"]) for row in rows)
    ivgtyp = np.ascontiguousarray(
        np.asarray([int(row["ivgtyp"]) for row in rows],
                   dtype=np.int32).reshape(1, -1))
    isltyp = np.ascontiguousarray(
        np.asarray([int(row["isltyp"]) for row in rows],
                   dtype=np.int32).reshape(1, -1))

    inputs = {field: _column(rows, column)
              for field, column in INPUT_COLUMNS.items()}
    for key in ("tile2_category","tile2_tsk","tile2_snow","tile2_snowh","w1","w2"):
        if key in rows[0]: inputs[key]=_column(rows,key)
    for field in ("smois", "tslb", "sh2o"):
        profile = np.empty((4, 1, len(rows)), dtype=np.float32)
        for layer in range(4):
            profile[layer] = _column(rows, f"{field}{layer + 1}_in")
        inputs[field] = np.ascontiguousarray(profile)

    present = set(rows[0])
    reference = {field: _column(rows, column)
                 for field, column in OUTPUT_COLUMNS.items()
                 if column in present}
    for field in ("flx4", "fvb", "fbur", "fgsn", "t2", "th2", "q2"):
        if field in rows[0]: reference[field] = _column(rows, field)
    soil_reference = {}
    for field in SOIL_OUTPUT_COLUMNS:
        profile = np.empty((4, 1, len(rows)), dtype=np.float32)
        for layer in range(4):
            profile[layer] = _column(rows, f"{field}{layer + 1}")
        soil_reference[field] = np.ascontiguousarray(profile)

    sfcdiags_reference = {field: _column(rows, column)
                          for field, column in SFCDIAGS_COLUMNS.items()
                          if column in present}

    return NoahOracleFixture(
        name=name, switches=dict(known[name]), cases=cases,
        ivgtyp=ivgtyp, isltyp=isltyp, inputs=inputs, reference=reference,
        soil_reference=soil_reference, sfcdiags_reference=sfcdiags_reference,
    )


def noah_port_outputs(fixture: NoahOracleFixture) -> dict[str, np.ndarray]:
    """Run ``launch_noah`` on the fixture and bring the results back to host.

    The kernel updates state in place, so the device arrays start as copies of
    the fixture's inputs -- which is what the WRF driver was handed too.
    """
    import cupy as cp

    from woof.core.noah import launch_noah, load_tables, pack_params

    params = pack_params(load_tables())
    device: dict[str, "cp.ndarray"] = {
        "ivgtyp": cp.asarray(fixture.ivgtyp),
        "isltyp": cp.asarray(fixture.isltyp),
    }
    for field, value in fixture.inputs.items():
        device[field] = cp.asarray(value.copy())
    shape = fixture.ivgtyp.shape
    for field in ("noahres", "reslin", "chklowq"):
        device[field] = cp.zeros(shape, dtype=cp.float32)
    device["smcrel"] = cp.zeros((4, *shape), dtype=cp.float32)
    device["ebal"] = cp.zeros(shape, dtype=cp.int32)

    if fixture.switches.get("ua_phys"):
        for name in ("flx4","fvb","fbur","fgsn"):
            device[name]=cp.zeros(shape,dtype=cp.float32)
    switches=dict(fixture.switches)
    mosaic=switches.pop("mosaic",False)
    ntiles=switches.pop("mosaic_cat",1)
    if mosaic:
        from woof.core.noah_mosaic import launch_noah_mosaic,MOSAIC_TILE_FIELDS,MOSAIC_SOIL_FIELDS,load_mosaic_categories
        for name in MOSAIC_TILE_FIELDS:
            device[name]=cp.ascontiguousarray(cp.stack([device[name.removesuffix("_mosaic")]]*ntiles))
        for name in MOSAIC_SOIL_FIELDS:
            device[name]=cp.ascontiguousarray(cp.concatenate([device[name.removesuffix("_mosaic")]]*ntiles))
        device["mosaic_cat_index"]=cp.ascontiguousarray(device["ivgtyp"][None,...])
        device["landusef2"]=cp.ones((1,*shape),dtype=cp.float32)
        if ntiles==2:
            for name in ("tsk","snow","snowh"):
                device[name+"_mosaic"][1]=device["tile2_"+name]
            device["mosaic_cat_index"]=cp.ascontiguousarray(cp.stack([device["ivgtyp"],device["tile2_category"].astype(cp.int32)]))
            device["landusef2"]=cp.ascontiguousarray(cp.stack([device["w1"],device["w2"]]))
        categories=load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH",isurban=13,isice=15,iswater=17)
        launch_noah_mosaic(device,params,FIXTURE_DT,FIXTURE_DZS,
            mosaic_cat=ntiles,categories=categories,xice_threshold=FIXTURE_XICE_THRESHOLD,
            itimestep=FIXTURE_ITIMESTEP,**switches)
    else:
        launch_noah(device, params, FIXTURE_DT, FIXTURE_DZS,
                    isurban=FIXTURE_ISURBAN, isice=FIXTURE_ISICE,
                    xice_threshold=FIXTURE_XICE_THRESHOLD,
                    itimestep=FIXTURE_ITIMESTEP, **switches)
    if "t2" in fixture.reference:
        from woof.core.noah_sfcdiags import refresh,refresh_ua
        from woof.core import constants as c
        for name in ("t2","th2","q2"): device[name]=cp.zeros(shape,dtype=cp.float32)
        if fixture.switches.get("ua_phys"):
            refresh_ua(device,device["qv1"],device["sfctmp"],rcp=c.RCP)
        else:
            refresh(device,device["qv1"],rd=c.RD,cp_air=c.CP,p0=c.P0,rcp=c.RCP,active_floor=1.e-5)
        for name in ("flx4","fvb","fbur","fgsn"): device.setdefault(name,cp.zeros(shape,dtype=cp.float32))
    return {name: cp.asnumpy(value) for name, value in device.items()
            if name not in ("ivgtyp", "isltyp", "ebal")}


def noah_sfcdiags_outputs(fixture: NoahOracleFixture,
                          port: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """T2, TH2 and Q2 from the port's SFCDIAGS on the port's own lsm outputs.

    The call is the one ``PhysicsDriver._refresh_surface_diagnostics`` makes
    after every Noah step (:func:`woof.core.noah_sfcdiags.refresh`, with the
    driver's constants), on the fields ``launch_noah`` wrote.  The fixture's
    reference ran WRF's SFCDIAGS on the words WRF's lsm wrote, so a word that
    differs here is SFCDIAGS's own or one inherited from the LSM, and the LSM
    words are graded separately.
    """
    import cupy as cp

    from woof.core import constants as c
    from woof.core import noah_sfcdiags

    f32 = np.float32
    fields = {name: cp.asarray(np.ascontiguousarray(port[name], f32))
              for name in ("tsk", "cqs2", "chs2", "qsfc", "qfx", "hfx")}
    fields["psfc"] = cp.asarray(fixture.inputs["psfc"])
    for name in SFCDIAGS_COLUMNS:
        fields[name] = cp.zeros(fixture.ivgtyp.shape, dtype=cp.float32)
    noah_sfcdiags.refresh(
        fields, cp.asarray(fixture.inputs["qv1"]), rd=f32(c.RD),
        cp_air=f32(c.CP), p0=f32(c.P0), rcp=f32(c.RCP),
        active_floor=f32(1.0e-5))
    return {name: cp.asnumpy(fields[name]) for name in SFCDIAGS_COLUMNS}


@dataclass(frozen=True)
class FieldParity:
    """The measurement for one output field."""

    field: str
    max_ulp: int
    differing: int
    total: int
    worst_case: int
    worst_layer: int | None
    got: float
    want: float

    def __str__(self) -> str:
        where = (f"case {self.worst_case}"
                 + ("" if self.worst_layer is None
                    else f" layer {self.worst_layer}"))
        return (f"{self.field:<10s} max_ulp={self.max_ulp:<12d} "
                f"differing={self.differing}/{self.total} "
                f"worst at {where}: got {self.got!r} want {self.want!r}")


def _worst(fixture, field, got, want, keep) -> FieldParity:
    got = np.ascontiguousarray(got, np.float32)[..., keep]
    want = np.ascontiguousarray(want, np.float32)[..., keep]
    kept_cases = [c for c, k in zip(fixture.cases, keep) if k]
    distance = fp32_ulp_distance(got, want)
    flat = int(np.argmax(distance)) if distance.size else 0
    index = np.unravel_index(flat, distance.shape) if distance.size else ()
    if got.ndim == 3:
        layer, _, col = index
        worst_layer = int(layer) + 1
    else:
        _, col = index
        worst_layer = None
    return FieldParity(
        field=field,
        max_ulp=int(distance.max()) if distance.size else 0,
        differing=int(np.count_nonzero(
            got.view(np.uint32) != want.view(np.uint32))),
        total=int(got.size),
        worst_case=kept_cases[int(col)] if kept_cases else -1,
        worst_layer=worst_layer,
        got=float(got.ravel()[flat]), want=float(want.ravel()[flat]),
    )


def measure_noah_parity(fixture: NoahOracleFixture,
                        port: dict[str, np.ndarray],
                        *, include_glacial: bool = True
                        ) -> dict[str, FieldParity]:
    """ULP distance from every port output to the word ``lsm`` wrote.

    ``include_glacial=False`` measures the ordinary-land columns alone, the
    population tests/test_noah_wrf461_parity.py's ULP table pins.
    """
    keep = np.ones(fixture.ncase, dtype=bool)
    if not include_glacial:
        keep &= ~fixture.glacial
    result: dict[str, FieldParity] = {}
    for field in fixture.reference:
        result[field] = _worst(fixture, field, port[field],
                               fixture.reference[field], keep)
    for field in SOIL_OUTPUT_COLUMNS:
        result[field] = _worst(fixture, field, port[field],
                               fixture.soil_reference[field], keep)
    return result

