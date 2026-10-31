"""WRF's urban parameter tables and urban land-use categories.

Transcribes ``urban_param_init`` (``phys/module_sf_urban.F:2043-2551``, WRF
v4.7.1, byte-identical to v4.6.1 for this routine) into :func:`load_urban_params`,
which returns a frozen :class:`UrbanParams` keyed by WRF's own names
(``ZR_TBL``, ``FRC_URB_TBL``, ``CH_SCHEME_DATA`` ...).  All three urban
models (UCM, BEP, BEP+BEM) read their tables through that one object.

Precision is WRF's ``-r4`` build: every table value is the binary32 nearest
the decimal text (gfortran's list-directed READ rounds once, correctly), and
every derived value (``HGT_TBL``, ``ZDC_TBL``, ``Z0C_TBL``, ``SVF_TBL`` ...) is
evaluated statement by statement in binary32 through WOOF's own ``expf`` and
``powf`` (:mod:`woof.core.noahmp_libm`), because the Fortran column oracle
the kernels are proven against was compiled that way.

Categories are TABLE ROWS, never literals: :func:`urban_category_set` reads
``NATURAL`` and ``LCZ_1..LCZ_11`` from the VEGPARM.TBL section the run's
land-use dataset names, and ``ISURBAN`` comes from the caller (the static
index or the Noah-MP table), exactly as WRF takes them.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.core.noahmp_libm import expf, f32, powf, sqrtf

URBAN_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "urban"

#: SHA-256 of the vendored tables (WRF v4.7.1 ``run/``; byte-identical to
#: v4.6.1).  :func:`load_urban_params` refuses any other bytes: a table that
#: moved is a different model, and the oracle fixtures were built from these.
URBAN_TABLE_SHA256 = MappingProxyType({
    "URBPARM.TBL":
        "5811226b3db503ae02d8b35cb5cb10e0e90804e451f0f4a5b76a274ee64773d0",
    "URBPARM_LCZ.TBL":
        "ab08e3f79d2f5d9d329aa2c953de4e7d94a71741baa0a50f1ecc145c6f81d9ab",
})

# Table generations are separate from runtime canopy implementations.
# WRF v4.6.1, v4.7.0 and v4.7.1 have the same two table byte streams.
URBAN_TABLE_GENERATIONS = MappingProxyType({
    "wrf_461": URBAN_TABLE_SHA256,
    "wrf_470": URBAN_TABLE_SHA256,
    "wrf_471": URBAN_TABLE_SHA256,
    "wrf_480": MappingProxyType({
        "URBPARM.TBL":
            "16db7a3ea930f28b02af32306a2a4261fdf3e3377f4be6e9ba4605a4b8269847",
        "URBPARM_LCZ.TBL":
            "89c396ae52071d6cd1b3ef729ae61799dd12c68e851b06e1dc361a7ea0183019",
    }),
})
_NBS_ROWS = ("FVG", "RTREE", "HTREE", "DTREE", "LAI_TREE", "LAI_VEG",
             "Z0VG", "Z0HVG", "EPSVG", "EPST", "ALBVG", "ALBT")
_NBS_SWITCHES = {"TREEOPTION": "TREEOPTION", "TTscheme": "TTscheme"}

#: module_sf_urban.F:70-72.
MAXDIRS = 3
MAXHGTS = 50
#: ``num_roof_layers = num_wall_layers = num_road_layers = num_soil_layers``
#: (module_sf_urban.F:2090-2092); Noah and Noah-MP both fix 4.
URBAN_LAYERS = 4

#: The per-category REAL arrays ``urban_param_init`` allocates, in its own
#: allocation order (module_sf_urban.F:2134-2244).
_CATEGORY_REALS = (
    "ZR", "SIGMA_ZED", "Z0C", "Z0HC", "ZDC", "SVF", "R", "RW", "HGT", "AH",
    "ALH", "BETR", "BETB", "BETG", "CAPR", "CAPB", "CAPG", "AKSR", "AKSB",
    "AKSG", "ALBR", "ALBB", "ALBG", "EPSR", "EPSB", "EPSG", "Z0R", "Z0B",
    "Z0G", "AKANDA_URBAN", "Z0HB", "Z0HG", "TRLEND", "TBLEND", "TGLEND",
    "FRC_URB", "COP", "BLDAC_FRC", "COOLED_FRC", "PWIN", "BETA", "TIME_ON",
    "TIME_OFF", "TARGTEMP", "GAPTEMP", "TARGHUM", "GAPHUM", "PERFLO", "HSESF",
    "PV_FRAC_ROOF", "GR_FRAC_ROOF")
_CATEGORY_INTS = ("SW_COND",)

#: Table key -> (WRF array name) for the per-category rows read verbatim.
_ROW_KEYS = {
    "ZR": "ZR", "SIGMA_ZED": "SIGMA_ZED", "AH": "AH", "ALH": "ALH",
    "FRC_URB": "FRC_URB", "ALBR": "ALBR", "ALBB": "ALBB", "ALBG": "ALBG",
    "EPSR": "EPSR", "EPSB": "EPSB", "EPSG": "EPSG",
    "AKANDA_URBAN": "AKANDA_URBAN", "Z0B": "Z0B", "Z0G": "Z0G",
    "TRLEND": "TRLEND", "TBLEND": "TBLEND", "TGLEND": "TGLEND", "Z0R": "Z0R",
    "COP": "COP", "BLDAC_FRC": "BLDAC_FRC", "COOLED_FRC": "COOLED_FRC",
    "PWIN": "PWIN", "BETA": "BETA", "TIME_ON": "TIME_ON",
    "TIME_OFF": "TIME_OFF", "TARGTEMP": "TARGTEMP", "GAPTEMP": "GAPTEMP",
    "TARGHUM": "TARGHUM", "GAPHUM": "GAPHUM", "PERFLO": "PERFLO",
    "HSEQUIP_SCALE_FACTOR": "HSESF", "PV_FRAC_ROOF": "PV_FRAC_ROOF",
    "GR_FRAC_ROOF": "GR_FRAC_ROOF",
}
#: The three heat-capacity rows, converted J m-3 K-1 -> cal cm-3 deg-1
#: (module_sf_urban.F:2275-2286), and the three conductivities,
#: J m-1 s-1 K-1 -> cal cm-1 s-1 deg-1 (:2287-2298).
_CAP_KEYS = ("CAPR", "CAPB", "CAPG")
_AKS_KEYS = ("AKSR", "AKSB", "AKSG")
#: Integer switches (``*_DATA`` and options), each a single list item.
_SCALAR_INTS = {
    "BOUNDR": "BOUNDR_DATA", "BOUNDB": "BOUNDB_DATA", "BOUNDG": "BOUNDG_DATA",
    "CH_SCHEME": "CH_SCHEME_DATA", "TS_SCHEME": "TS_SCHEME_DATA",
    "AHOPTION": "AHOPTION", "ALHOPTION": "ALHOPTION",
    "IMP_SCHEME": "IMP_SCHEME", "IRI_SCHEME": "IRI_SCHEME",
    "GROPTION": "GROPTION", "GR_FLAG": "GR_FLAG_TBL", "GR_TYPE": "GR_TYPE_TBL",
}
_SCALAR_REALS = {"OASIS": "OASIS", "FGR": "FGR"}
#: Fixed-length REAL arrays: key -> (WRF name, length).
_FIXED_REALS = {
    "AHDIUPRF": ("AHDIUPRF", 24), "ALHSEASON": ("ALHSEASON", 4),
    "ALHDIUPRF": ("ALHDIUPRF", 48), "PORIMP": ("PORIMP", 3),
    "DENGIMP": ("DENGIMP", 3), "DZGR": ("DZGR", 4),
    "HSEQUIP": ("HSEQUIP_TBL", 24), "IRHO": ("IRHO_TBL", 24),
}
#: The layer thicknesses, read in metres and converted to cm (:2311-2322).
_LAYER_KEYS = {"DDZR": "DZR", "DDZB": "DZB", "DDZG": "DZG"}

_F32_ONE_OVER_CAL = f32(f32(1.0) / f32(4.1868))


class UrbanTableError(ValueError):
    """``urban_param_init``'s FATAL_ERROR arms, and the refused table arms."""


def decimal_to_f32(token: str) -> np.float32:
    """The binary32 nearest a Fortran REAL literal, rounded ONCE.

    gfortran's list-directed READ converts decimal text straight to
    ``REAL(4)`` with correct rounding; ``np.float32(float(text))`` rounds
    twice (to binary64, then to binary32) and can land one ULP off for text
    near a binary32 midpoint.  This picks the nearest of the two binary32
    neighbours of the double rounding, ties to even, against the exact
    rational value of the text.
    """
    text = token.strip().replace("d", "e").replace("D", "e")
    exact = Fraction(text)
    candidate = np.float32(float(exact))
    below = np.nextafter(candidate, np.float32(-np.inf), dtype=np.float32)
    above = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    best = candidate
    best_err = abs(Fraction(float(candidate)) - exact)
    for other in (below, above):
        if not np.isfinite(other):
            continue
        err = abs(Fraction(float(other)) - exact)
        if err < best_err or (err == best_err
                              and int(other.view(np.uint32)) % 2 == 0):
            best, best_err = other, err
    return np.float32(best)


def _items(text: str) -> list[str]:
    """Fortran list-directed items of one internal record.

    Separators are commas and blanks; ``r*c`` repeat counts are expanded; a
    ``/`` ends the record.  URBPARM.TBL uses none of the last two, but they
    are what the READ would do, so they are what this does.
    """
    out: list[str] = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        if tok.startswith("/"):
            break
        if "*" in tok:
            count, value = tok.split("*", 1)
            out.extend([value] * int(count))
        else:
            out.append(tok)
    return out


def _read_reals(text: str, count: int, name: str) -> np.ndarray:
    items = _items(text)
    if len(items) < count:
        raise UrbanTableError(
            f"URBPARM row {name!r} carries {len(items)} values, the READ "
            f"needs {count}: gfortran's list-directed READ would fail at the "
            "end of the record")
    return np.array([decimal_to_f32(v) for v in items[:count]],
                    dtype=np.float32)


def _read_ints(text: str, count: int, name: str) -> np.ndarray:
    items = _items(text)
    if len(items) < count:
        raise UrbanTableError(
            f"URBPARM row {name!r} carries {len(items)} values, the READ "
            f"needs {count}")
    return np.array([int(v) for v in items[:count]], dtype=np.int32)


@dataclass(frozen=True)
class UrbanParams:
    """``urban_param_init``'s module state, frozen.

    ``values`` maps WRF's names to NumPy arrays: per-category arrays are
    ``<NAME>_TBL`` of length ``ICATE`` (index ``utype - 1``); the street and
    height blocks are ``(MAXDIRS, ICATE)`` / ``(MAXHGTS, ICATE)`` in the
    Fortran dimension order; ``DZR``/``DZB``/``DZG`` are the four layer
    thicknesses in centimetres.  Scalars are 0-d arrays.  Attribute access
    (``params.ZR_TBL``) reads the same mapping.
    """

    table: str
    sf_urban_physics: int
    use_wudapt_lcz: int
    icate: int
    sha256: str
    values: Mapping[str, np.ndarray] = field(repr=False)
    generation: str = "wrf_471"

    def __getattr__(self, name: str):
        values = object.__getattribute__(self, "values")
        try:
            return values[name]
        except KeyError:
            raise AttributeError(name) from None

    def scalar(self, name: str):
        value = self.values[name]
        return value.item() if value.ndim == 0 else value


def urban_table_path(use_wudapt_lcz: int, tbl_dir: Path | None = None) -> Path:
    name = "URBPARM_LCZ.TBL" if int(use_wudapt_lcz) else "URBPARM.TBL"
    return (Path(tbl_dir) if tbl_dir is not None else URBAN_TABLE_DIR) / name


def load_urban_params(sf_urban_physics: int, use_wudapt_lcz: int = 0, *,
                      tbl_dir: Path | None = None,
                      verify_sha256: bool = True,
                      generation: str = "wrf_471") -> UrbanParams:
    """``urban_param_init`` (module_sf_urban.F:2043-2551), transcribed.

    ``sf_urban_physics`` matters: option 1 rederives ``Z0R_TBL`` from the
    roof-height spread (:2505-2512); options 2/3 keep the table's row.
    ``slucm_distributed_drag`` is not an argument because woof refuses it at
    the namelist door (its arm is not transcribed).
    """
    if generation not in URBAN_TABLE_GENERATIONS:
        raise UrbanTableError(f"unknown urban table generation {generation!r}; "
                              f"choose {tuple(URBAN_TABLE_GENERATIONS)}")
    if generation == "wrf_480" and tbl_dir is None:
        tbl_dir = URBAN_TABLE_DIR / "wrf480"
    nbs = generation == "wrf_480"
    pins = URBAN_TABLE_GENERATIONS[generation]
    row_keys = {**_ROW_KEYS, **({k: k for k in _NBS_ROWS} if nbs else {})}
    scalar_ints = {**_SCALAR_INTS, **(_NBS_SWITCHES if nbs else {})}
    cap_keys = (*_CAP_KEYS, "CAPVG") if nbs else _CAP_KEYS
    option = int(sf_urban_physics)
    if option not in (1, 2, 3):
        raise ValueError(f"load_urban_params: sf_urban_physics={option} runs "
                         "no urban model")
    path = urban_table_path(use_wudapt_lcz, tbl_dir)
    from woof.tables import table_path
    family = "urban-lcz" if use_wudapt_lcz else "urban"
    if generation == "wrf_480":
        family += "-480"
    if verify_sha256:
        from woof.tables import CorruptTableAssets, MissingTableAssets
        try:
            path = table_path(family, path)
        except (CorruptTableAssets, MissingTableAssets) as error:
            raise UrbanTableError(str(error)) from error
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_sha256 and digest != pins[path.name]:
        raise UrbanTableError(
            f"{path} has SHA-256 {digest}, not the {generation} "
            f"table's {pins[path.name]}: the urban oracle "
            "fixtures were built from the pinned bytes, so another table is "
            "an unproven model")
    lines = raw.decode("ascii").split("\n")
    v: dict[str, np.ndarray] = {}
    icate = 0
    roof_width = road_width = None
    dz = {}
    numdir = numhgt = None
    i = 0
    while i < len(lines):
        # read(11,'(A512)') keeps trailing blanks; a CR would be a byte of
        # the record, but the vendored tables are LF-only (hash-pinned).
        string = lines[i]
        i += 1
        if string[:1] == "#":
            continue
        if string.strip() == "":
            continue
        indx = string.find(":")
        if indx < 0:
            continue
        name = string[:indx].strip()
        rest = string[indx + 1:]
        if name == "Number of urban categories":
            icate = int(_items(rest)[0])
            for arr in _CATEGORY_REALS:
                v.setdefault(f"{arr}_TBL",
                             np.full(icate, np.nan, dtype=np.float32))
            for arr in _CATEGORY_INTS:
                v.setdefault(f"{arr}_TBL", np.zeros(icate, dtype=np.int32))
            # :2250-2256, reset on every categories line.
            numdir = np.zeros(icate, dtype=np.int32)
            v["STREET_DIRECTION_TBL"] = np.full((MAXDIRS, icate), -1.0e36,
                                                dtype=np.float32)
            v["STREET_WIDTH_TBL"] = np.zeros((MAXDIRS, icate),
                                             dtype=np.float32)
            v["BUILDING_WIDTH_TBL"] = np.zeros((MAXDIRS, icate),
                                               dtype=np.float32)
            numhgt = np.zeros(icate, dtype=np.int32)
            v["HEIGHT_BIN_TBL"] = np.full((MAXHGTS, icate), -1.0e36,
                                          dtype=np.float32)
            v["HPERCENT_BIN_TBL"] = np.full((MAXHGTS, icate), -1.0e36,
                                            dtype=np.float32)
        elif icate == 0 and name != "Number of urban categories":
            raise UrbanTableError(
                f"URBPARM row {name!r} precedes 'Number of urban "
                "categories': WRF would index an unallocated array")
        elif name == "ROOF_WIDTH":
            roof_width = _read_reals(rest, icate, name)
        elif name == "ROAD_WIDTH":
            road_width = _read_reals(rest, icate, name)
        elif name in row_keys:
            v[f"{row_keys[name]}_TBL"] = _read_reals(rest, icate, name)
        elif name in cap_keys:
            row = _read_reals(rest, icate, name)
            v[f"{name}_TBL"] = np.array(
                [f32(f32(float(x) * _F32_ONE_OVER_CAL) * f32(1.0e-6))
                 for x in row], dtype=np.float32)
        elif name in _AKS_KEYS:
            row = _read_reals(rest, icate, name)
            v[f"{name}_TBL"] = np.array(
                [f32(f32(float(x) * _F32_ONE_OVER_CAL) * f32(1.0e-2))
                 for x in row], dtype=np.float32)
        elif name == "SW_COND":
            v["SW_COND_TBL"] = _read_ints(rest, icate, name)
        elif name in _LAYER_KEYS:
            row = _read_reals(rest, URBAN_LAYERS, name)
            dz[_LAYER_KEYS[name]] = np.array(
                [f32(float(x) * f32(100.0)) for x in row], dtype=np.float32)
        elif name in scalar_ints:
            v[scalar_ints[name]] = np.array(_read_ints(rest, 1, name)[0],
                                             dtype=np.int32)
        elif name in _SCALAR_REALS:
            v[_SCALAR_REALS[name]] = np.array(_read_reals(rest, 1, name)[0],
                                              dtype=np.float32)
        elif name in _FIXED_REALS:
            target, count = _FIXED_REALS[name]
            v[target] = _read_reals(rest, count, name)
        elif name == "STREET PARAMETERS":
            # :2345-2355.  No iostat test inside the loop: a table that ends
            # before END STREET PARAMETERS is a READ error in WRF too.
            while True:
                if i >= len(lines):
                    raise UrbanTableError("STREET PARAMETERS block never ends")
                string = lines[i]
                i += 1
                if string[:1] == "#" or string.strip() == "":
                    continue
                if string.rstrip() == "END STREET PARAMETERS":
                    break
                items = _items(string)
                k = int(items[0])
                numdir[k - 1] += 1
                n = int(numdir[k - 1])
                v["STREET_DIRECTION_TBL"][n - 1, k - 1] = decimal_to_f32(items[1])
                v["STREET_WIDTH_TBL"][n - 1, k - 1] = decimal_to_f32(items[2])
                v["BUILDING_WIDTH_TBL"][n - 1, k - 1] = decimal_to_f32(items[3])
        elif name == "BUILDING HEIGHTS":
            k = int(_items(rest)[0])
            while True:
                if i >= len(lines):
                    raise UrbanTableError("BUILDING HEIGHTS block never ends")
                string = lines[i]
                i += 1
                if string[:1] == "#" or string.strip() == "":
                    continue
                if string.rstrip() == "END BUILDING HEIGHTS":
                    break
                items = _items(string)
                numhgt[k - 1] += 1
                n = int(numhgt[k - 1])
                v["HEIGHT_BIN_TBL"][n - 1, k - 1] = decimal_to_f32(items[0])
                v["HPERCENT_BIN_TBL"][n - 1, k - 1] = decimal_to_f32(items[1])
            column = v["HPERCENT_BIN_TBL"][:, k - 1]
            # sum(..., mask=(>-1.E25)): gfortran accumulates in order in
            # REAL(4).
            pctsum = 0.0
            for value in column:
                if value > np.float32(-1.0e25):
                    pctsum = f32(pctsum + float(value))
            if pctsum != 100.0:
                raise UrbanTableError(
                    f"Building height percentages for category {k} must sum "
                    f"to 100.0, they sum to {pctsum:.2f} (urban_param_init's "
                    "'pctsum is not equal to 100.' fatal)")
        else:
            raise UrbanTableError(
                f'URBPARM.TBL: Unrecognized NAME = "{name}" in Subr '
                "URBAN_PARAM_INIT (WRF's own fatal)")
    if icate == 0:
        raise UrbanTableError(f"{path} names no urban categories")
    if roof_width is None or road_width is None:
        raise UrbanTableError(
            "URBPARM table lacks ROOF_WIDTH or ROAD_WIDTH: urban_param_init "
            "divides by them and then deallocates both, so WRF itself fails")
    v["ICATE"] = np.array(icate, dtype=np.int32)
    v["NUMDIR_TBL"] = numdir
    v["NUMHGT_TBL"] = numhgt
    for name in ("DZR", "DZB", "DZG"):
        if name not in dz:
            raise UrbanTableError(f"URBPARM table lacks D{name}")
        v[name] = dz[name]
    if nbs:
        required = [*(f"{name}_TBL" for name in (*_NBS_ROWS, "CAPVG")),
                    *_NBS_SWITCHES.values()]
        missing = [name for name in required if name not in v]
        if missing:
            raise UrbanTableError(
                f"wrf_480 table lacks {missing}: read_param would read "
                "uninitialized nature-based solution table state")
        for name in _NBS_SWITCHES.values():
            if int(v[name]) not in (0, 1):
                raise UrbanTableError(
                    f"{name} must be 0 or 1: the WRF tree solver has no "
                    f"temperature or canopy arm for {int(v[name])}")
    _derive(v, icate, roof_width, road_width, option, nbs=nbs)
    if nbs:
        if int(v["TREEOPTION"]) == 1:
            for lc in range(icate):
                # VF_TREE_ANALYTICAL:5431-5434. Equality is allowed.
                if (f32(2.0 * float(v["RTREE_TBL"][lc])) > v["RW_TBL"][lc]
                        or f32(float(v["HTREE_TBL"][lc])
                               + float(v["RTREE_TBL"][lc])) > v["HGT_TBL"][lc]):
                    raise UrbanTableError(
                        f"category {lc+1}: Invalid tree geometry in "
                        "VF_TREE_ANALYTICAL: 2*rt must be <= W and ht + rt <= H")
    _refuse_untranscribed_arms(v, path)
    frozen = {name: _frozen(value) for name, value in v.items()}
    return UrbanParams(table=path.name, sf_urban_physics=option,
                       use_wudapt_lcz=int(use_wudapt_lcz), icate=icate,
                       sha256=digest, values=MappingProxyType(frozen),
                       generation=generation)


def _frozen(value: np.ndarray) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _derive(v: dict, icate: int, roof_width, road_width, option: int, *,
            nbs: bool = False) -> None:
    """module_sf_urban.F:2463-2546: the values urban_param_init computes."""
    cd = f32(1.2)
    alpha_macd = f32(4.43)
    beta_macd = f32(1.0)
    vonk = f32(0.4)
    vonk2 = f32(vonk * vonk)
    half = f32(0.5)
    v["Z0HB_TBL"] = np.array([f32(f32(0.1) * float(x)) for x in v["Z0B_TBL"]],
                             dtype=np.float32)
    v["Z0HG_TBL"] = np.array([f32(f32(0.1) * float(x)) for x in v["Z0G_TBL"]],
                             dtype=np.float32)
    for name in ("HGT", "R", "RW", "BETR", "BETB", "BETG", "ZDC", "Z0C",
                 "Z0HC", "SVF"):
        v[f"{name}_TBL"] = v[f"{name}_TBL"].copy()
    z0r = v["Z0R_TBL"].copy()
    if nbs:
        v["RTREE_M_TBL"] = v["RTREE_TBL"].copy()
        v["RW_M_TBL"] = road_width.copy()
    for lc in range(icate):
        zr = float(v["ZR_TBL"][lc])
        road = float(road_width[lc])
        roof = float(roof_width[lc])
        width = f32(road + roof)
        hgt = f32(zr / width)
        r = f32(roof / width)
        rw = f32(1.0 - r)
        v["HGT_TBL"][lc] = hgt
        v["R_TBL"][lc] = r
        if nbs:
            # module_sf_urban.F:3451-3459, inside the source category loop.
            for name in ("RTREE", "DTREE", "HTREE"):
                v[f"{name}_TBL"][lc] = f32(float(v[f"{name}_TBL"][lc]) / width)
        v["RW_TBL"][lc] = rw
        v["BETR_TBL"][lc] = 0.0
        v["BETB_TBL"][lc] = 0.0
        v["BETG_TBL"][lc] = 0.0
        lambda_p = r
        lambda_f = hgt
        # ZDC = ZR * (1.0 + (alpha_macd ** (-Lambda_P)) * (Lambda_P - 1.0))
        term = f32(float(powf(alpha_macd, f32(-lambda_p)))
                   * f32(lambda_p - 1.0))
        zdc = f32(zr * f32(1.0 + term))
        v["ZDC_TBL"][lc] = zdc
        one_minus = f32(1.0 - f32(zdc / zr))
        coef = f32(f32(f32(half * beta_macd) * cd) / vonk2)
        # Z0C = ZR*(1-ZDC/ZR)*exp(-(0.5*b*Cd/VonK**2*(1-ZDC/ZR)*Lambda_F)**(-0.5))
        inner = f32(f32(coef * one_minus) * lambda_f)
        z0c = f32(f32(zr * one_minus)
                  * float(expf(f32(-float(powf(inner, f32(-0.5)))))))
        v["Z0C_TBL"][lc] = z0c
        if option == 1:
            lambda_fr = f32(float(v["SIGMA_ZED_TBL"][lc]) / width)
            inner_r = f32(f32(coef * one_minus) * lambda_fr)
            z0r[lc] = f32(f32(zr * one_minus)
                          * float(expf(f32(-float(powf(inner_r,
                                                        f32(-0.5)))))))
        v["Z0HC_TBL"][lc] = f32(f32(0.1) * z0c)
        # Sky view factor, :2527-2545.
        dhgt = f32(hgt / 100.0)
        h = f32(hgt - f32(dhgt / 2.0))
        vfws = 0.0
        for _ in range(99):
            h = f32(h - dhgt)
            root = float(sqrtf(f32(f32(h * h) + f32(rw * rw))))
            vfws = f32(vfws + f32(0.25 * f32(1.0 - f32(h / root))))
        vfws = f32(vfws / 99.0)
        vfws = f32(vfws * 2.0)
        vfgs = f32(1.0 - f32(f32(f32(2.0 * vfws) * hgt) / rw))
        v["SVF_TBL"][lc] = vfgs
    v["Z0R_TBL"] = z0r


#: Table switches whose non-default arm no woof kernel transcribes, and the
#: value WRF's vendored tables carry.  A table that selects another value
#: would have that row silently ignored, so it is refused.
UNTRANSCRIBED_TABLE_ARMS = MappingProxyType({
    "OASIS": (1.0, "the urban oasis evaporation factor passed to Noah's SFLX "
                   "as AOASIS (module_sf_noahdrv.F:996-1033)"),
    "IRI_SCHEME": (0, "urban irrigation of the natural fraction "
                      "(module_sf_noahdrv.F:1003-1017)"),
})


def _refuse_untranscribed_arms(v: dict, path: Path) -> None:
    for name, (admitted, what) in UNTRANSCRIBED_TABLE_ARMS.items():
        if name in v and float(v[name]) != float(admitted):
            raise UrbanTableError(
                f"{path.name} sets {name}={float(v[name])!r}: {what} is not "
                "transcribed in this build, so the row would be ignored")


# ---------------------------------------------------------------------------
# categories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UrbanCategories:
    """The land-use categories WRF treats as urban, and NATURAL.

    ``lcz`` is ``(LCZ_1, ..., LCZ_11)`` as VEGPARM.TBL numbers them for the
    dataset (51-61 since WRF 4.4.2).
    """

    isurban: int
    natural: int
    lcz: tuple[int, ...]

    @property
    def urban(self) -> tuple[int, ...]:
        return (self.isurban, *self.lcz)

    def utype(self, category: int, use_wudapt_lcz: int) -> int:
        """``urban_var_init``'s UTYPE_URB2D (module_sf_urban.F:2748-2786)."""
        category = int(category)
        if category == self.isurban:
            return 5 if int(use_wudapt_lcz) else 2
        for n, lcz in enumerate(self.lcz, start=1):
            if category == lcz:
                return n
        return 0

    def utype_lookup(self, use_wudapt_lcz: int) -> np.ndarray:
        """int32 array indexed by category number -> UTYPE (0 = not urban)."""
        size = max((*self.urban, self.natural)) + 1
        table = np.zeros(size, dtype=np.int32)
        # LCZ first, ISURBAN last: urban_var_init's IF chain tests ISURBAN
        # first, so a table that gave ISURBAN an LCZ number resolves to it.
        for n, lcz in enumerate(self.lcz, start=1):
            table[lcz] = n
        table[self.isurban] = 5 if int(use_wudapt_lcz) else 2
        return table


def urban_category_set(mminlu: str = "MODIFIED_IGBP_MODIS_NOAH", *,
                       isurban: int,
                       tbl_dir: Path | None = None) -> UrbanCategories:
    """``NATURAL`` and ``LCZ_1..LCZ_11`` from VEGPARM.TBL, plus ``isurban``.

    ``SOIL_VEG_GEN_PARM`` (module_sf_noahdrv.F) reads them as label/value
    record pairs after the category rows of the dataset's section.
    """
    from woof.core.noah import TBL_DIR, _tokens

    d = Path(tbl_dir) if tbl_dir is not None else TBL_DIR
    from woof.tables import table_path
    lines = table_path("noah", d / "VEGPARM.TBL").read_text().splitlines()
    start = None
    for idx, line in enumerate(lines):
        toks = _tokens(line)
        if toks and toks[0].strip("'") == mminlu:
            start = idx
            break
    if start is None:
        raise ValueError(f"land-use dataset {mminlu!r} not in VEGPARM.TBL")
    found: dict[str, int] = {}
    wanted = {"NATURAL", *(f"LCZ_{n}" for n in range(1, 12))}
    idx = start + 1
    while idx < len(lines) - 1 and len(found) < len(wanted):
        label = lines[idx].strip()
        if label.startswith("Vegetation Parameters"):
            break
        if label in wanted:
            found[label] = int(_tokens(lines[idx + 1])[0])
            idx += 2
            continue
        idx += 1
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"VEGPARM.TBL section {mminlu!r} lacks {missing}")
    return UrbanCategories(
        isurban=int(isurban), natural=found["NATURAL"],
        lcz=tuple(found[f"LCZ_{n}"] for n in range(1, 12)))


def lcz_categories(mminlu: str = "MODIFIED_IGBP_MODIS_NOAH", *,
                   tbl_dir: Path | None = None) -> tuple[int, ...]:
    """``LCZ_1..LCZ_11`` as VEGPARM.TBL numbers them for ``mminlu``."""
    return urban_category_set(mminlu, isurban=-1, tbl_dir=tbl_dir).lcz


def ucm_canopy_heights(use_wudapt_lcz: int, *, generation: str = "wrf_471",
                      tbl_dir=None) -> dict[int, float]:
    """``ZDC + Z0C + 2`` m per urban class of the table the UCM would read.

    ``module_sf_urban.F:825`` stops the model (``FATAL_ERROR``) on any
    urban cell where ``ZDC+Z0C+2. >= ZA``, ``ZA`` being the first mass
    level's height (``0.5*DZ8W(1)``, noahdrv.F:818 / noahmpdrv.F:3404).
    ``ZDC`` and ``Z0C`` come from the table row of the cell's class
    (:2509-2512 derive them at ``urban_param_init``), so the height each
    class needs is known before any cell is read.  Summed in float32 in
    WRF's order.  Keys are urban types (``utype``, 1-based).
    """
    params = load_urban_params(1, int(use_wudapt_lcz),
                               generation=generation, tbl_dir=tbl_dir)
    zdc, z0c = params.values["ZDC_TBL"], params.values["Z0C_TBL"]
    return {utype + 1: float(f32(f32(zdc[utype] + z0c[utype]) + f32(2.0)))
            for utype in range(params.icate)}


__all__ = [
    "MAXDIRS", "MAXHGTS", "URBAN_LAYERS", "URBAN_TABLE_DIR",
    "URBAN_TABLE_SHA256", "URBAN_TABLE_GENERATIONS",
    "UNTRANSCRIBED_TABLE_ARMS", "UrbanCategories",
    "UrbanParams", "UrbanTableError", "decimal_to_f32", "lcz_categories",
    "load_urban_params",
    "ucm_canopy_heights", "urban_category_set", "urban_table_path",
]
