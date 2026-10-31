"""Noah land surface model: parameter tables and the column-kernel launcher.

The physics is transcribed line-faithfully from the bundle's WRF v4.6.1
``phys/module_sf_noahdrv.F`` (subroutine ``lsm``: the per-column input
prep and the post-``SFLX`` state/flux updates) and
``phys/module_sf_noahlsm.F`` (``SFLX`` and its full subtree).  One CUDA
thread integrates one land column (kernels/noah.cu); the float64
verification mirror is :func:`woof.verify.npref.np_noah_column`.

Scope (Phase 3 Task 10): 4 soil layers, snowpack (SWE/depth/density,
melt, cover fraction, snow albedo aging), canopy water, the full surface
energy balance producing TSK/HFX/QFX/LH/GRDFLX, runoff, and the WRF
driver diagnostics (SMSTAV/SMSTOT/SMCREL/ACSNOM/ACSNOW/SNOPCX/POTEVP/
NOAHRES), and the land-ice column: ``ivgtyp == isice`` runs WRF's
``SFLX_GLACIAL`` (module_sf_noahlsm_glacial_only.F) through the driver's
ICE == -1 arm in ``noah_glacial_column``, as WRF does (until 2.9.0 such a
column was left untouched), graded bitwise against WRF by
tests/test_noah_wrf461_parity.py.  NOT ported: UA_PHYS, FASDAS, WRF-Hydro,
the urban canopy models themselves (they run after the kernel,
:mod:`woof.core.urban_driver`; the kernel carries their hand-over through
``launch_noah(urban=...)``, and the plain ``VEGTYP==ISURBAN`` parameter
overrides inside SFLX/HRT ARE ported) and ``SFCDIF_off`` (the exchange
coefficient CH comes from the surface-layer scheme).

Parameter tables: VEGPARM.TBL / SOILPARM.TBL / GENPARM.TBL are shipped
verbatim from the WRF v4.6.1 ``run/`` directory in
``woof/data/noah_tables`` (see PROVENANCE.md there) and parsed here by
transcribing the list-directed READ sequence of ``SOIL_VEG_GEN_PARM``
(module_sf_noahdrv.F).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TBL_DIR = Path(__file__).resolve().parent.parent / "data" / "noah_tables"

#: Noah's soil geometry, stated once, beside the scheme that owns it.
#:
#: WRF resolves the soil-layer count from ``sf_surface_physics`` rather than
#: honouring the namelist request: ``share/module_check_a_mundo.F`` subroutine
#: ``set_physics_rconfigs`` sets ``num_soil_layers = 4`` for LSMSCHEME (:3553)
#: and for NOAHMPSCHEME (:3555), and both schemes take their layer depths from
#: the *same* generator, ``init_soil_depth_2``
#: (``share/module_soil_pre.F:795`` for Noah, ``:807`` for Noah-MP).  That
#: shared generator -- not a coincidence of two literals -- is why
#: :data:`woof.config.LAND_SURFACE_SOIL_LAYERS` sources the Noah-MP entry
#: from this constant.  ``init_soil_depth_2`` itself carries WRF's fatal for
#: any other request ("The Noah and NoahMP LSMs use 4 layers.",
#: ``share/module_soil_pre.F:1140-1143``), so 4 is not a woof choice.
#: ``init_soil_depth_2`` emits dzs, whose accumulation is the layer-bound
#: table in :mod:`woof.ingest.soil_contract`; that module checks itself
#: against this count so the two cannot drift.
NUM_SOIL_LAYERS = 4
#: ``init_soil_depth_2``'s dzs (share/module_soil_pre.F:1138).
SOIL_LAYER_THICKNESS_M = (0.10, 0.30, 0.60, 1.00)

# Packed-table column layouts shared by the CUDA kernel and the mirror.
VEG_COLS = ("nroot", "rsmin", "rgl", "hs", "snup", "laimin", "laimax",
            "emissmin", "emissmax", "albedomin", "albedomax",
            "z0min", "z0max", "shdtbl", "maxalb")
SOIL_COLS = ("bexp", "smcdry", "f1", "smcmax", "smcref", "psisat",
             "dksat", "dwsat", "smcwlt", "quartz")
GEN = {name: i for i, name in enumerate(
    ("topt", "cmcmax", "cfactr", "rsmax", "sbeta", "fxexp", "csoil",
     "salp", "refdk", "refkdt", "frzk", "zbot", "lvcoef", "slope",
     "bare", "natural"))}

_TPB = 64          # threads per block (one thread per land column)


# ---------------------------------------------------------------------------
# table parsing (SOIL_VEG_GEN_PARM transcription)
# ---------------------------------------------------------------------------

def _tokens(line: str) -> list[str]:
    """Fortran list-directed items of one record: comma/space separated,
    quoted strings kept whole."""
    return [match.group(0) for match in re.finditer(r"'[^']*'|[^,\s]+", line)]


@dataclass
class NoahTables:
    """Raw parsed tables (float64), one attribute per Fortran table array."""
    # VEGPARM
    lutype: str = ""
    lucats: int = 0
    shdtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nrotbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rstbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rgltbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hstbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    snuptbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    maxalb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    laimintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    laimaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    emissmintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    emissmaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    albedomintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    albedomaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z0mintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z0maxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ztopvtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zbotvtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    topt: float = 0.0
    cmcmax: float = 0.0
    cfactr: float = 0.0
    rsmax: float = 0.0
    bare: int = 0
    natural: int = 0
    # SOILPARM
    sltype: str = ""
    slcats: int = 0
    bb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drysmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f11: np.ndarray = field(default_factory=lambda: np.zeros(0))
    maxsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    refsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satpsi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satdk: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satdw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wltsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qtz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # GENPARM
    slope_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sbeta: float = 0.0
    fxexp: float = 0.0
    csoil: float = 0.0
    salp: float = 0.0
    refdk: float = 0.0
    refkdt: float = 0.0
    frzk: float = 0.0
    zbot: float = 0.0
    czil: float = 0.0
    smlow: float = 0.0
    smhigh: float = 0.0
    lvcoef: float = 0.0


def _parse_table_columns(lines, cats, names, table):
    cols = {n: np.zeros(cats) for n in names}
    for _ in range(cats):
        toks = _tokens(next(lines))
        lc = int(float(toks[0])) - 1
        for n, tok in zip(names, toks[1:1 + len(names)]):
            cols[n][lc] = float(tok)
    for n in names:
        setattr(table, n, cols[n])


def _parse_vegparm(text: str, mminlu: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    next(lines)                                     # header line
    while True:                                     # FIND_LUTYPE
        lutype = _tokens(next(lines))[0].strip("'")
        cats_line = _tokens(next(lines))
        lucats = int(cats_line[0])
        if lutype == mminlu:
            break
        # skip to the next 'Vegetation Parameters' flag line
        for line in lines:
            if line.startswith("Vegetation Parameters"):
                break
        else:
            raise ValueError(f"landuse dataset {mminlu!r} not found "
                             f"in VEGPARM.TBL")
    t.lutype, t.lucats = lutype, lucats
    names = ("shdtbl nrotbl rstbl rgltbl hstbl snuptbl maxalb laimintbl "
             "laimaxtbl emissmintbl emissmaxtbl albedomintbl albedomaxtbl "
             "z0mintbl z0maxtbl ztopvtbl zbotvtbl").split()
    _parse_table_columns(lines, lucats, names, t)
    # scalar block: label line then value line, in the READ order
    def val():
        next(lines)
        return _tokens(next(lines))[0]
    t.topt = float(val())
    t.cmcmax = float(val())
    t.cfactr = float(val())
    t.rsmax = float(val())
    t.bare = int(val())
    t.natural = int(val())
    # the Fortran skips two records (CROP label + value) then checks the
    # next record is not a new section before the LCZ block; the LCZ
    # indices are urban-model-only and unused here.


def _parse_soilparm(text: str, mminsl: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    while True:
        line = next(lines)                          # 'Soil Parameters'
        sltype = _tokens(next(lines))[0].strip("'")
        cats = int(_tokens(next(lines))[0])
        if sltype == mminsl:
            break
        for _ in range(cats):                       # skip this section
            next(lines)
    t.sltype, t.slcats = sltype, cats
    names = "bb drysmc f11 maxsmc refsmc satpsi satdk satdw wltsmc qtz".split()
    _parse_table_columns(lines, cats, names, t)


def _parse_genparm(text: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    next(lines)                                     # 'General Parameters'
    next(lines)                                     # 'SLOPE_DATA'
    n = int(_tokens(next(lines))[0])
    t.slope_data = np.array([float(_tokens(next(lines))[0])
                             for _ in range(n)])
    for name in ("sbeta", "fxexp", "csoil", "salp", "refdk", "refkdt",
                 "frzk", "zbot", "czil", "smlow", "smhigh", "lvcoef"):
        next(lines)                                 # label record
        setattr(t, name, float(_tokens(next(lines))[0]))


def load_tables(mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
                mminsl: str = "STAS",
                tbl_dir: Path | None = None) -> NoahTables:
    """Parse the shipped WRF run-directory tables (SOIL_VEG_GEN_PARM)."""
    d = Path(tbl_dir) if tbl_dir is not None else TBL_DIR
    from woof.tables import table_path
    t = NoahTables()
    _parse_vegparm(table_path("noah", d / "VEGPARM.TBL").read_text(), mminlu, t)
    _parse_soilparm(table_path("noah", d / "SOILPARM.TBL").read_text(), mminsl, t)
    _parse_genparm(table_path("noah", d / "GENPARM.TBL").read_text(), t)
    return t


# ---------------------------------------------------------------------------
# packing for the kernel / mirror
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoahParams:
    """Packed float64 parameter tables (rows = category-1; columns per
    VEG_COLS / SOIL_COLS / GEN)."""
    veg: np.ndarray
    soil: np.ndarray
    gen: np.ndarray
    lucats: int
    slcats: int
    bare: int
    natural: int
    lutype: str
    sltype: str


#: (ISURBAN, ISICE) for each land-use dataset VEGPARM.TBL carries.  WRF's
#: Noah driver takes both from the model configuration, which
#: share/input_wrf.F fills from the input file's global attributes (geogrid
#: writes them from its land-use index) and, failing that, from MMINLU
#: (input_wrf.F:849-862: MODIS 15, otherwise 24 for ISICE).  A table and not
#: a default: until 2026-10-12 the driver passed the MODIS pair (13, 15) for
#: every dataset, so a USGS run applied the urban overrides to category 13
#: (evergreen broadleaf) and SKIPPED category 15 (mixed forest) as land ice.
NOAH_LANDUSE_CATEGORY_IDS = {
    "MODIFIED_IGBP_MODIS_NOAH": (13, 15),
    "MODI-RUC": (13, 15),
    "NLCD40": (13, 15),
    "USGS": (1, 24),
    "USGS-RUC": (1, 24),
}


def noah_category_ids(params: "NoahParams") -> tuple[int, int]:
    """``(isurban, isice)`` of the dataset ``params`` was packed from."""
    try:
        return NOAH_LANDUSE_CATEGORY_IDS[params.lutype]
    except KeyError:
        raise ValueError(
            f"Noah has no ISURBAN/ISICE identity for land-use dataset "
            f"{params.lutype!r}: the column would apply the urban overrides "
            "and the land-ice routine to whatever categories 13 and 15 mean "
            "in that dataset.  Add the dataset's pair to "
            "woof.core.noah.NOAH_LANDUSE_CATEGORY_IDS") from None


def pack_params(t: NoahTables) -> NoahParams:
    veg = np.zeros((t.lucats, len(VEG_COLS)))
    src = dict(nroot=t.nrotbl, rsmin=t.rstbl, rgl=t.rgltbl, hs=t.hstbl,
               snup=t.snuptbl, laimin=t.laimintbl, laimax=t.laimaxtbl,
               emissmin=t.emissmintbl, emissmax=t.emissmaxtbl,
               albedomin=t.albedomintbl, albedomax=t.albedomaxtbl,
               z0min=t.z0mintbl, z0max=t.z0maxtbl, shdtbl=t.shdtbl,
               maxalb=t.maxalb)
    for i, name in enumerate(VEG_COLS):
        veg[:, i] = src[name]
    soil = np.zeros((t.slcats, len(SOIL_COLS)))
    ssrc = dict(bexp=t.bb, smcdry=t.drysmc, f1=t.f11, smcmax=t.maxsmc,
                smcref=t.refsmc, psisat=t.satpsi, dksat=t.satdk,
                dwsat=t.satdw, smcwlt=t.wltsmc, quartz=t.qtz)
    for i, name in enumerate(SOIL_COLS):
        soil[:, i] = ssrc[name]
    gen = np.zeros(len(GEN))
    for name, i in GEN.items():
        if name == "slope":
            gen[i] = t.slope_data[0]     # driver: SLOPETYP = 1 always
        elif name == "bare":
            gen[i] = t.bare
        elif name == "natural":
            gen[i] = t.natural
        else:
            gen[i] = getattr(t, name)
    return NoahParams(veg=veg, soil=soil, gen=gen, lucats=t.lucats,
                      slcats=t.slcats, bare=t.bare, natural=t.natural,
                      lutype=t.lutype, sltype=t.sltype)


def noah_initial_snow_albedo(
        geogrid_percent, ivgtyp, params: NoahParams, *, rdmaxalb: bool):
    """Transcribe ``module_sf_noahdrv.F:1902-1904``'s LSMINIT choice.

    ``rdmaxalb=True`` keeps the supplied geogrid SNOALB percentage.  False
    replaces it with VEGPARM's MAXALB percentage for the one-based vegetation
    category.  The returned fraction is what woof's Noah kernel consumes.
    """
    supplied = np.asarray(geogrid_percent, dtype=np.float64)
    categories = np.asarray(ivgtyp)
    if supplied.shape != categories.shape:
        raise ValueError(
            "SNOALB and IVGTYP shapes differ at Noah initialization")
    if bool(rdmaxalb):
        selected = supplied
    else:
        if (not np.issubdtype(categories.dtype, np.integer)
                or np.any(categories < 1)
                or np.any(categories > params.lucats)):
            raise ValueError(
                "IVGTYP is outside the Noah VEGPARM category table")
        selected = params.veg[categories.astype(np.int64) - 1,
                              VEG_COLS.index("maxalb")]
    return np.asarray(selected * 0.01, dtype=np.float32)


# ---------------------------------------------------------------------------
# LSMINIT soil-liquid-water helper (CPU, float64)
# ---------------------------------------------------------------------------

def noah_frh2o(tkelv: float, smc: float, sh2o: float, smcmax: float,
                bexp: float, psis: float) -> float:
    """Scalar FRH2O qualification authority with its original input types.

    Runtime soil arrays use sh2o_init and its Rust field entry. This scalar
    authority retains the host reference for qualification and standalone prep.
    """
    ck, blim, error = 8.0, 5.5, 0.005
    hlice, gs, t0 = 3.335e5, 9.81, 273.15
    bx = bexp if bexp <= blim else blim
    nlog = 0
    kcount = 0
    if tkelv > (t0 - 1.0e-3):
        return smc
    swl = smc - sh2o
    if swl > (smc - 0.02):
        swl = smc - 0.02
    if swl < 0.0:
        swl = 0.0
    while (nlog < 10) and (kcount == 0):
        nlog += 1
        df = (math.log((psis * gs / hlice) * ((1.0 + ck * swl) ** 2.0)
                       * (smcmax / (smc - swl)) ** bx)
              - math.log(-(tkelv - t0) / tkelv))
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        if swlk > (smc - 0.02):
            swlk = smc - 0.02
        if swlk < 0.0:
            swlk = 0.0
        dswl = abs(swlk - swl)
        swl = swlk
        if dswl <= error:
            kcount += 1
    free = smc - swl
    if kcount == 0:
        fk = (((hlice / (gs * (-psis))) * ((tkelv - t0) / tkelv))
              ** (-1.0 / bx)) * smcmax
        if fk < 0.02:
            fk = 0.02
        free = min(fk, smc)
    return free


def sh2o_init(smois, tslb, isltyp, params: NoahParams) -> np.ndarray:
    """SH2O from SMOIS/TSLB exactly as LSMINIT (module_sf_noahdrv.F):
    Flerchinger explicit first guess, then the FRH2O Newton iteration."""
    smois = np.asarray(smois, np.float64)
    tslb = np.asarray(tslb, np.float64)
    if smois.shape != tslb.shape or smois.ndim == 0:
        raise ValueError("smois and tslb must be same-shape soil profiles")
    column_shape = smois.shape[1:]
    soil_type = np.asarray(isltyp)
    try:
        soil_type = np.broadcast_to(soil_type, column_shape)
    except ValueError as exc:
        raise ValueError("isltyp must match the soil-profile columns") from exc
    from woof.noah_init_bridge import initialize
    indices = [SOIL_COLS.index(name) for name in ("bexp", "smcmax", "psisat")]
    table = params.soil[:params.slcats, indices]
    return initialize(smois, tslb, soil_type, table)


# ---------------------------------------------------------------------------
# kernel launcher
# ---------------------------------------------------------------------------

# ordered 2-D real field names, matching the noah_column kernel signature
_F2D = ("psfc", "sfcprs", "sfctmp", "qv1", "qgh", "dz8w1", "glw",
        "swdown", "rainbl", "sr", "chs", "cqs2", "chs2", "rib",
        "vegfra", "shdmin", "shdmax", "tmn", "xland", "xice", "snoalb",
        "embck",
        "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc",
        "canwat", "snow", "snowc", "snowh",
        "albedo", "albbck", "emiss", "znt", "z0", "snotime", "lai",
        "smstav", "smstot",
        "sfcrunoff", "udrunoff", "acsnow", "acsnom", "snopcx", "potevp",
        "noahres", "reslin", "chklowq")
_F3D = ("smois", "tslb", "sh2o", "smcrel")


def _device_tables(params: NoahParams, dzs):
    """The four CONSTANT kernel inputs, uploaded once per parameter set.

    ``launch_noah`` used to rebuild these on every call: four
    ``cp.asarray`` of host NumPy -- the vegetation table, the soil table,
    the general parameters and the soil-layer thicknesses -- for a kernel
    that only ever reads them.  On a 15 s step that is four pageable H2D
    copies per model step for data that is fixed at load time.

    Two things it costs, and the second is the one that mattered here.  The
    copies themselves are small but pageable, so each is a staged,
    synchronising transfer.  And an H2D of any kind is illegal inside a CUDA
    graph capture, so this single line was what stopped every rung from
    ``+Noah LSM`` upward from being capturable -- see the census in
    :mod:`tilestream.graphcap`.

    The cache lives in this module and NOT on the ``params`` object.
    Attaching it to ``params`` was the first attempt and
    ``woof/io/restart.py`` was right to refuse the equivalent on the
    radiation callable: an array attribute on a driver object is state a
    restart must account for, and a cached constant is not state.

    It is keyed by the values it uploads, not by the identity of
    ``params``.  An identity key held a reference to every parameter object
    it had seen, and each forecast run in one process loads its own: every
    member of an ensemble left one more copy of the same tables on the card
    for the rest of the run (tests/test_ensemble_member_release_gpu.py).  By
    value, every forecast with the same tables shares one upload, a forecast
    with other tables gets its own, and nothing keeps a finished forecast's
    parameter object alive.
    """
    import cupy as cp

    from woof.core.device_cache import cached_ready

    host = (params.veg.astype(np.float32).ravel(),
            params.soil.astype(np.float32).ravel(),
            params.gen.astype(np.float32),
            np.asarray(dzs, np.float32))
    key = (int(cp.cuda.Device().id),
           *((table.shape, table.tobytes()) for table in host))
    def upload():
        return tuple(cp.asarray(table) for table in host)
    return cached_ready(cp, _DEVICE_TABLES, key, upload)


#: ``(device, the uploaded tables by value) -> device tables``.  See :func:`_device_tables`.
_DEVICE_TABLES: dict = {}

def initialize_noah_liquid_water(dev: dict, params: NoahParams, dzs) -> None:
    """Run WRF LSMINIT's cold-start liquid-water initialization on CUDA.

    Call after native fields have been restored and before loading any
    checkpoint. Warm soil copies SMOIS exactly; frozen soil uses the
    existing Noah FRH2O device function and its FP32 arithmetic.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    moisture, temperature, liquid = (dev[name] for name in ("smois", "tslb", "sh2o"))
    if moisture.ndim != 3 or moisture.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError("Noah initialization requires four soil layers")
    for name, value in (("smois", moisture), ("tslb", temperature), ("sh2o", liquid)):
        if (not isinstance(value, cp.ndarray) or value.shape != moisture.shape
                or value.dtype != cp.float32 or not value.flags.c_contiguous):
            raise ValueError(f"{name}: expected contiguous float32 soil profiles")
    soil = dev["isltyp"]
    if (not isinstance(soil, cp.ndarray) or soil.shape != moisture.shape[1:]
            or soil.dtype != cp.int32 or not soil.flags.c_contiguous):
        raise ValueError("isltyp: expected contiguous int32 soil categories")
    if bool(cp.any((soil < 1) | (soil > params.slcats)).item()):
        raise ValueError("isltyp category is outside the Noah soil table")
    _, soil_table, _, _ = _device_tables(params, dzs)
    columns = soil.size
    kernel = get_kernel("noah_init", "noah_initialize_sh2o")
    kernel(((columns + _TPB - 1) // _TPB,), (_TPB,),
           (soil, moisture, temperature, liquid, soil_table,
            np.int32(params.slcats), np.int32(columns)))

#: ``noah_seaice_column``'s 2-D arguments, in kernel order.
SEAICE_F2D = ("xice", "psfc", "sfcprs", "sfctmp", "qv1", "qgh", "glw",
              "swdown", "rainbl", "sr", "rib", "chs", "cqs2", "chs2", "tsk",
              "emiss", "albedo", "z0", "snow", "snowc", "snowh", "znt", "lh",
              "hfx", "qfx", "potevp", "grdflx", "qsfc", "acsnow", "acsnom",
              "snopcx", "sfcrunoff", "noahres")

#: WRF v4.6.1 Registry.EM_COMMON defaults for the sea-ice namelist, the only
#: values woof runs: seaice_albedo_opt = 0, seaice_thickness_opt = 0,
#: seaice_snowdepth_opt = 0 and fractional_seaice = 0.
SEAICE_THICKNESS_DEFAULT_M = 3.0
SEAICE_SNOWDEPTH_MIN_M = 0.001
SEAICE_SNOWDEPTH_MAX_M = 1.0e10


def launch_noah_seaice(dev: dict, dt: float, *, frpcpn: bool,
                       xice_threshold: float = 0.5,
                       albedo_default: float = 0.65,
                       check_snow_density: bool = False) -> None:
    """WRF's ``seaice_noah`` on device arrays (in place).

    The surface driver calls it after ``lsm`` / ``lsm_mosaic`` on every Noah
    run (module_surface_driver.F:2917-2929); it advances every column with
    ``xice >= xice_threshold`` -- skin temperature, snow on ice, the four-layer
    ice temperature profile and the surface fluxes -- and leaves every other
    column untouched.  ``dt`` is the MODEL time step: the driver hands
    seaice_noah ``dt`` where it hands lsm ``dtbl``.

    ``check_snow_density`` reads back WRF's one fatal (SFLX_SEAICE's
    "Physical snow depth is less than snow water equiv."), which only an
    inconsistent initial SNOW/SNOWH pair can reach: once the scheme has run,
    SNOWPACK holds the density at or below 0.4.  The forecast asks for it on
    its first surface call, so the read costs one synchronisation per run.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel
    from woof.core.state import DTYPE

    ny, nx = dev["tsk"].shape
    args = []
    for name in SEAICE_F2D:
        a = dev[name]
        if a.shape != (ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"sea ice {name}: expected ({ny},{nx}) float32")
        args.append(a)
    tslb = dev["tslb"]
    if tslb.shape != (NUM_SOIL_LAYERS, ny, nx) or tslb.dtype != cp.float32:
        raise ValueError(f"sea ice tslb: expected ({NUM_SOIL_LAYERS},{ny},{nx}) float32")
    fatal = cp.zeros((1,), dtype=cp.int32)
    args += [tslb, fatal, DTYPE(dt), DTYPE(xice_threshold),
             np.int32(1 if frpcpn else 0), DTYPE(albedo_default),
             DTYPE(SEAICE_THICKNESS_DEFAULT_M), DTYPE(SEAICE_SNOWDEPTH_MIN_M),
             DTYPE(SEAICE_SNOWDEPTH_MAX_M), np.int32(ny), np.int32(nx)]
    kern = get_kernel("noah", "noah_seaice_column")
    kern(((ny * nx + _TPB - 1) // _TPB,), (_TPB,), tuple(args))
    if check_snow_density and int(fatal.get()[0]) != 0:
        raise ValueError(
            "Noah sea ice: Physical snow depth is less than snow water equiv. "
            "(WRF SFLX_SEAICE's fatal): a sea-ice column has SNOW/1000 > SNOWH "
            "after the 0.01 m floor and the depth clamp, so its snow density "
            "exceeds 1 and the ice-column conduction would be computed from "
            "an impossible snowpack.  Fix the initial SNOW/SNOWH pair.")


def _urban_args(urban, ny: int, nx: int) -> list:
    """The ten trailing ``noah_column_urban`` arguments for the urban
    handover.  The default path (``urban is None``) launches
    ``noah_column``, which takes none of them."""
    import cupy as cp

    option = int(urban["option"])
    if option not in (1, 2, 3):
        raise ValueError(f"urban handover for sf_urban_physics={option}")
    mask = urban["category_mask"]
    if mask.dtype != cp.int32 or mask.ndim != 1:
        raise ValueError("urban category_mask must be a 1-D int32 array")
    checked = []
    for name in ("frc_urb2d", "ts_urb2d", "tsk_rural_bep", "rural_q1",
                 "rural_q2k", "rural_zlvl"):
        a = urban.get(name)
        if a is None and name == "tsk_rural_bep" and option == 1:
            a = urban["ts_urb2d"]          # never read at urban_opt == 1
        if a is None or a.shape != (ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"urban {name}: expected ({ny},{nx}) float32")
        checked.append(a)
    return [np.int32(option), mask, np.int32(mask.size),
            np.int32(urban["natural"]), *checked]


#: module_sf_noahdrv.F:443 ``TRESH=.95E0`` (single precision).
NOAH_CHKLOWQ_TRESH = np.float32(0.95)


def noah_chklowq_myj(dev: dict) -> None:
    """The ``myj`` arm of Noah's CHKLOWQ (module_sf_noahdrv.F:805-815), in place.

    WRF's ``lsm`` writes CHKLOWQ for EVERY column, before its land/water
    branch::

        Q2K   = QV3D(i,1,j)/(1.0+QV3D(i,1,j))
        Q2SAT = QGH(I,J)/(1.0+QGH(I,J))
        IF ((myj) .AND. (Q2K .GE. Q2SAT*TRESH) .AND. Q2K .LT. QZ0(I,J)) THEN
           CHKLOWQ(I,J) = 0.
        ELSE
           CHKLOWQ(I,J) = 1.

    with ``myj`` true when ``sf_sfclay_physics`` is MYJ (2) or QNSE (4)
    (module_surface_driver.F:1899-1900).  ``noah_column`` writes the ELSE
    value (1) for every column; under the Eta surface layer this sets the
    WRF value.  The breakage it prevents: the MYJ PBL scales the surface
    latent flux by CHKLOWQ (module_bl_myjpbl.F:548, :576), so near-saturated
    land columns whose QZ0 exceeds the first-level humidity kept a moisture
    flux WRF switches off.  QV3D, QGH and QZ0 are all ``lsm`` inputs it does
    not write, so evaluating the test after the column kernel is the same
    arithmetic.  Each operation is its own float32 kernel, so no
    multiply-add contraction can enter, in either numerics mode.
    """
    import cupy as cp

    one = np.float32(1.0)
    qv1 = dev["qv1"]
    qgh = dev["qgh"]
    q2k = qv1 / (one + qv1)
    q2sat = qgh / (one + qgh)
    low = (q2k >= q2sat * NOAH_CHKLOWQ_TRESH) & (q2k < dev["qz0"])
    cp.copyto(dev["chklowq"], np.float32(0.0), where=low)


def launch_noah(dev: dict, params: NoahParams, dt: float, dzs,
                isurban: int | None = None, isice: int | None = None,
                xice_threshold: float = 0.5, frpcpn: bool = False,
                usemonalb: bool = False, rdlai2d: bool = False,
                opt_thcnd: int = 1, itimestep: int = 2,
                urban: dict | None = None) -> None:
    """Run the ``noah_column`` kernel on device arrays (in place).

    ``dev`` maps field names to CuPy arrays: every name in ``_F2D`` as
    ``(ny, nx)`` float32 (``ivgtyp``/``isltyp`` int32), every name in
    ``_F3D`` as ``(4, ny, nx)`` float32, plus ``ebal`` (ny, nx) int32.
    Water (``xland >= 1.5``) and sea-ice (``xice >= xice_threshold``)
    columns are skipped exactly as the WRF driver skips them; land-ice
    (``ivgtyp == isice``) columns run SFLX_GLACIAL, as the driver's
    ICE == -1 arm does.

    ``urban`` is the urban model's handover
    (:meth:`woof.core.urban_driver.UrbanCoupler.noah_kernel_args`), or
    None -- the default path, which launches ``noah_column`` (the kernel
    without the hand-over compiled in) and passes no urban argument.  With it, urban columns run SFLX as WRF's NATURAL
    category from the rural skin temperature (module_sf_noahdrv.F:964-990),
    BEP/BEM's ``tsk_rural_bep`` is updated after SFLX (:1243-1247), and
    ``q1``/``q2k``/``zlvl`` are written for the blend.
    """
    import cupy as cp

    from woof.core.kernels import get_kernel
    from woof.core.state import DTYPE

    ny, nx = dev["tsk"].shape
    if len(dzs) != NUM_SOIL_LAYERS:
        raise ValueError(
            f"Noah port is fixed at {NUM_SOIL_LAYERS} soil layers")
    if (isinstance(itimestep, bool) or not isinstance(itimestep, int)
            or itimestep < 1):
        raise ValueError("Noah itimestep must be a positive integer")
    args = [dev["ivgtyp"], dev["isltyp"]]
    for name in _F2D:
        a = dev[name]
        if a.shape != (ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected ({ny},{nx}) float32")
        args.append(a)
    for name in _F3D:
        a = dev[name]
        if a.shape != (4, ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected (4,{ny},{nx}) float32")
        args.append(a)
    args.append(dev["ebal"])
    vegtbl, soiltbl, genp, dzs4 = _device_tables(params, dzs)
    # The dataset's own category identities unless the caller names them
    # (an oracle fixture does): WRF's driver receives ISURBAN and ISICE.
    if isurban is None or isice is None:
        dataset_isurban, dataset_isice = noah_category_ids(params)
        isurban = dataset_isurban if isurban is None else isurban
        isice = dataset_isice if isice is None else isice
    args += [vegtbl, soiltbl, genp, dzs4,
             DTYPE(dt), np.int32(params.lucats), np.int32(params.slcats),
             np.int32(isurban), np.int32(isice), DTYPE(xice_threshold),
             np.int32(itimestep),
             np.int32(1 if frpcpn else 0), np.int32(1 if usemonalb else 0),
             np.int32(1 if rdlai2d else 0), np.int32(opt_thcnd),
             np.int32(ny), np.int32(nx)]
    # The default path launches noah_column, the URBAN = false
    # instantiation, which compiles from exactly the statements it had
    # before the urban hand-over: compiling the hand-over into the one
    # kernel behind a runtime flag changed NVRTC's FMA contraction and moved
    # default forecasts (kernels/noah.cu).
    if urban is None:
        kern = get_kernel("noah", "noah_column")
    else:
        args += _urban_args(urban, ny, nx)
        kern = get_kernel("noah", "noah_column_urban")
    blocks = (ny * nx + _TPB - 1) // _TPB
    kern((blocks,), (_TPB,), tuple(args))
