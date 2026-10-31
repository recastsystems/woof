"""Byte-pinned Noah-MP v4.6.1 parameter-table loader.

The runtime solver is still under port. This module establishes the exact
table bytes and parsed parameter identity that every CPU/CUDA column oracle
and restart will consume; it does not make selector 4 executable by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from woof.core.noahmp_mynn_contract import NOAHMP_PARAMETER_ASSETS


ParameterAtom = bool | int | float | str

MPTABLE_GROUPS = (
    "noahmp_usgs_veg_categories",
    "noahmp_usgs_parameters",
    "noahmp_modis_veg_categories",
    "noahmp_modis_parameters",
    "noahmp_rad_parameters",
    "noahmp_global_parameters",
    "noahmp_irrigation_parameters",
    "noahmp_crop_parameters",
    "noahmp_tiledrain_parameters",
    "noahmp_optional_parameters",
)
SOIL_PARAMETER_NAMES = (
    "BB", "DRYSMC", "F11_OR_HC", "MAXSMC", "REFSMC", "SATPSI",
    "SATDK", "SATDW", "WLTSMC", "QTZ", "BVIC", "AXAJ", "BXAJ",
    "XXAJ", "BDVIC", "BBVIC", "GDVIC",
)
DATASET_GROUPS = MappingProxyType({
    "USGS": ("noahmp_usgs_veg_categories", "noahmp_usgs_parameters"),
    "MODIFIED_IGBP_MODIS_NOAH": (
        "noahmp_modis_veg_categories", "noahmp_modis_parameters",
    ),
})


@dataclass(frozen=True)
class NoahmpNamelistGroup:
    name: str
    values: Mapping[str, tuple[ParameterAtom, ...]]

    def scalar(self, name: str) -> ParameterAtom:
        values = self.values[name.upper()]
        if len(values) != 1:
            raise ValueError(f"{self.name}.{name} has {len(values)} values")
        return values[0]


@dataclass(frozen=True)
class NoahmpSoilRow:
    category: int
    values: tuple[float, ...]
    description: str


@dataclass(frozen=True)
class NoahmpSoilTable:
    name: str
    rows: tuple[NoahmpSoilRow, ...]

    def column(self, name: str) -> tuple[float, ...]:
        index = SOIL_PARAMETER_NAMES.index(name.upper())
        return tuple(row.values[index] for row in self.rows)


@dataclass(frozen=True)
class NoahmpGeneralTable:
    slopes: tuple[float, ...]
    values: Mapping[str, float]


@dataclass(frozen=True)
class NoahmpParameterBundle:
    mptable: Mapping[str, NoahmpNamelistGroup]
    soil: Mapping[str, NoahmpSoilTable]
    general: NoahmpGeneralTable
    receipt: Mapping[str, object]

    def vegetation_groups(
        self, dataset_identifier: str,
    ) -> tuple[NoahmpNamelistGroup, NoahmpNamelistGroup]:
        try:
            category_name, parameter_name = DATASET_GROUPS[dataset_identifier]
        except KeyError as exc:
            raise ValueError(
                "Noah-MP vegetation dataset must be USGS or "
                "MODIFIED_IGBP_MODIS_NOAH"
            ) from exc
        categories = self.mptable[category_name]
        parameters = self.mptable[parameter_name]
        nveg = categories.scalar("NVEG")
        if type(nveg) is not int or nveg < 1:
            raise ValueError(f"invalid {category_name}.NVEG={nveg!r}")
        for name, values in parameters.values.items():
            if len(values) not in (1, nveg):
                raise ValueError(
                    f"{parameter_name}.{name} has {len(values)} values; "
                    f"expected scalar or NVEG={nveg}"
                )
        return categories, parameters


def _strip_fortran_comment(line: str) -> str:
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "!":
            return line[:index]
    return line


def _split_fortran_values(text: str) -> list[str]:
    values: list[str] = []
    start = 0
    quote = ""
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == ",":
            token = text[start:index].strip()
            if token:
                values.append(token)
            start = index + 1
    if quote:
        raise ValueError("unterminated quoted MPTABLE value")
    token = text[start:].strip()
    if token:
        values.append(token)
    return values


def _parse_fortran_atom(token: str) -> ParameterAtom:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    lowered = token.lower()
    if lowered == ".true.":
        return True
    if lowered == ".false.":
        return False
    normalized = token.replace("D", "E").replace("d", "e")
    if any(char in normalized for char in ".eE"):
        return float(normalized)
    return int(normalized)


def _parse_fortran_values(text: str) -> tuple[ParameterAtom, ...]:
    output: list[ParameterAtom] = []
    for token in _split_fortran_values(text):
        repeated = re.fullmatch(r"([1-9][0-9]*)\*(.+)", token)
        if repeated:
            output.extend(
                [_parse_fortran_atom(repeated.group(2).strip())]
                * int(repeated.group(1))
            )
        else:
            output.append(_parse_fortran_atom(token))
    if not output:
        raise ValueError("empty MPTABLE assignment")
    return tuple(output)


def parse_mptable(text: str) -> Mapping[str, NoahmpNamelistGroup]:
    """Parse the restricted Fortran namelist syntax used by MPTABLE.TBL."""

    groups: dict[str, NoahmpNamelistGroup] = {}
    name: str | None = None
    values: dict[str, tuple[ParameterAtom, ...]] = {}
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = _strip_fortran_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("&"):
            if name is not None:
                raise ValueError(f"nested MPTABLE group at line {lineno}")
            name = line[1:].strip().lower()
            if not name or name in groups:
                raise ValueError(f"invalid or duplicate MPTABLE group at line {lineno}")
            values = {}
            continue
        if line == "/":
            if name is None:
                raise ValueError(f"orphan MPTABLE terminator at line {lineno}")
            groups[name] = NoahmpNamelistGroup(
                name=name, values=MappingProxyType(dict(values))
            )
            name = None
            values = {}
            continue
        if name is None or "=" not in line:
            raise ValueError(f"invalid MPTABLE line {lineno}: {raw_line!r}")
        key, payload = line.split("=", 1)
        key = key.strip().upper()
        if not key or key in values:
            raise ValueError(f"invalid or duplicate MPTABLE key at line {lineno}")
        values[key] = _parse_fortran_values(payload)
    if name is not None:
        raise ValueError(f"unterminated MPTABLE group {name}")
    if tuple(groups) != MPTABLE_GROUPS:
        raise ValueError(
            f"MPTABLE group inventory {tuple(groups)!r}; expected {MPTABLE_GROUPS!r}"
        )
    return MappingProxyType(groups)


def _parse_soil_row(line: str, *, section: str) -> NoahmpSoilRow:
    match = re.fullmatch(r"\s*([0-9]+),\s*(.*?)\s*,\s*'([^']+)'\s*", line)
    if not match:
        raise ValueError(f"invalid {section} SOILPARM row: {line!r}")
    values = tuple(
        float(token.strip().replace("D", "E").replace("d", "e"))
        for token in match.group(2).split(",")
    )
    if len(values) != len(SOIL_PARAMETER_NAMES):
        raise ValueError(
            f"{section} soil row has {len(values)} values; "
            f"expected {len(SOIL_PARAMETER_NAMES)}"
        )
    return NoahmpSoilRow(int(match.group(1)), values, match.group(3))


def parse_soilparm(text: str) -> Mapping[str, NoahmpSoilTable]:
    lines = text.splitlines()
    cursor = 0
    tables: dict[str, NoahmpSoilTable] = {}
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        if lines[cursor].strip() != "Soil Parameters":
            raise ValueError(f"unexpected SOILPARM title {lines[cursor]!r}")
        if cursor + 2 >= len(lines):
            raise ValueError("truncated SOILPARM section")
        section = lines[cursor + 1].strip()
        header = re.fullmatch(r"\s*([0-9]+),[0-9]+\s+'([^']+)'\s*", lines[cursor + 2])
        if not header:
            raise ValueError(f"invalid {section} SOILPARM header")
        count = int(header.group(1))
        header_names = tuple(header.group(2).split())
        if len(header_names) != len(SOIL_PARAMETER_NAMES):
            raise ValueError(f"invalid {section} SOILPARM field inventory")
        start = cursor + 3
        rows = tuple(
            _parse_soil_row(lines[start + index], section=section)
            for index in range(count)
        )
        if tuple(row.category for row in rows) != tuple(range(1, count + 1)):
            raise ValueError(f"nonsequential {section} SOILPARM categories")
        if section in tables:
            raise ValueError(f"duplicate SOILPARM section {section}")
        tables[section] = NoahmpSoilTable(section, rows)
        cursor = start + count
    if tuple(tables) != ("STAS", "STAS-RUC"):
        raise ValueError(f"SOILPARM section inventory {tuple(tables)!r}")
    return MappingProxyType(tables)


def parse_genparm(text: str) -> NoahmpGeneralTable:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5 or lines[:2] != ["General Parameters", "SLOPE_DATA"]:
        raise ValueError("invalid GENPARM header")
    slope_count = int(lines[2])
    slopes = tuple(float(value) for value in lines[3:3 + slope_count])
    remainder = lines[3 + slope_count:]
    if len(remainder) % 2:
        raise ValueError("GENPARM key/value inventory is not paired")
    values: dict[str, float] = {}
    for index in range(0, len(remainder), 2):
        key = remainder[index].upper()
        if key in values:
            raise ValueError(f"duplicate GENPARM key {key}")
        values[key] = float(remainder[index + 1].replace("D", "E").replace("d", "e"))
    return NoahmpGeneralTable(slopes, MappingProxyType(values))


def _read_pinned_table(root: Path, filename: str) -> tuple[str, dict[str, object]]:
    asset = next(
        item for item in NOAHMP_PARAMETER_ASSETS
        if Path(item.relative_path).name == filename
    )
    path = root / filename
    from woof.tables import table_path
    path = table_path("noahmp", path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != asset.bytes or digest != asset.sha256:
        raise ValueError(
            f"Noah-MP table {path} identity mismatch: bytes={len(payload)} "
            f"SHA-256={digest}; expected bytes={asset.bytes} "
            f"SHA-256={asset.sha256}"
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Noah-MP table {path} is not canonical ASCII") from exc
    return text, {
        "filename": filename, "bytes": len(payload), "sha256": digest,
    }


def load_noahmp_parameters(root: str | Path | None = None) -> NoahmpParameterBundle:
    """Load and byte-validate the packaged WRF v4.6.1 Noah-MP tables."""

    table_root = (
        Path(root) if root is not None
        else Path(__file__).parents[1] / "data" / "noahmp"
    )
    texts: dict[str, str] = {}
    assets: list[dict[str, object]] = []
    for filename in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        text, receipt = _read_pinned_table(table_root, filename)
        texts[filename] = text
        assets.append(receipt)
    bundle = NoahmpParameterBundle(
        mptable=parse_mptable(texts["MPTABLE.TBL"]),
        soil=parse_soilparm(texts["SOILPARM.TBL"]),
        general=parse_genparm(texts["GENPARM.TBL"]),
        receipt=MappingProxyType({
            "schema": "gpuwm.noahmp-parameters/v1",
            "status": "PASS",
            "root": str(table_root.resolve()),
            "assets": tuple(MappingProxyType(item) for item in assets),
        }),
    )
    bundle.vegetation_groups("USGS")
    bundle.vegetation_groups("MODIFIED_IGBP_MODIS_NOAH")
    return bundle


# ---------------------------------------------------------------------------
# TRANSFER_MP_PARAMETERS
#
# Transcribed from the pinned WRF v4.6.1 source
# ``phys/module_sf_noahmpdrv.F`` lines 1435-1710 together with the
# ``NOAHMP_TABLES`` readers in ``phys/module_sf_noahmplsm.F`` lines
# 11389-11893 and 12251-12307, which decide which namelist/table entry lands
# in each ``*_TABLE`` array. Nothing here is a physics choice: every value is
# either a table lookup or one of the two arithmetic expressions WRF forms in
# the transfer itself.
# ---------------------------------------------------------------------------

DEFAULT_VEGETATION_DATASET = "MODIFIED_IGBP_MODIS_NOAH"

_MONTHS = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
# read_mp_veg_parameters stores NROOT in a REAL table (module_sf_noahmplsm.F
# line 11172) that TRANSFER_MP_PARAMETERS assigns to an INTEGER component, so
# it truncates toward zero exactly like the Fortran assignment.
_VEGETATION_INTEGER_FIELDS = MappingProxyType({"NROOT": "NROOT"})
# parameters%FIELD = FIELD_TABLE(VEGTYPE): the table name equals the namelist
# key for all of these except RSMIN, which WRF reads from the MPTABLE key RS.
_VEGETATION_SCALAR_FIELDS = MappingProxyType({
    "CH2OP": "CH2OP", "DLEAF": "DLEAF", "Z0MVT": "Z0MVT", "HVT": "HVT",
    "HVB": "HVB", "DEN": "DEN", "RC": "RC", "MFSNO": "MFSNO",
    "SCFFAC": "SCFFAC", "CBIOM": "CBIOM", "SLA": "SLA", "DILEFC": "DILEFC",
    "DILEFW": "DILEFW", "FRAGR": "FRAGR", "LTOVRC": "LTOVRC",
    "C3PSN": "C3PSN", "KC25": "KC25", "AKC": "AKC", "KO25": "KO25",
    "AKO": "AKO", "VCMX25": "VCMX25", "AVCMX": "AVCMX", "BP": "BP",
    "MP": "MP", "QE25": "QE25", "AQE": "AQE", "RMF25": "RMF25",
    "RMS25": "RMS25", "RMR25": "RMR25", "ARM": "ARM", "FOLNMX": "FOLNMX",
    "TMIN": "TMIN", "XL": "XL", "MRP": "MRP", "CWPVT": "CWPVT",
    "WRRAT": "WRRAT", "WDPOOL": "WDPOOL", "TDLEF": "TDLEF", "RGL": "RGL",
    "RSMIN": "RS", "HS": "HS", "TOPT": "TOPT", "RSMAX": "RSMAX",
})
# The reader packs these from per-band MPTABLE keys into (VEGTYPE, band)
# tables; band 1 = visible, band 2 = near-infrared.
_VEGETATION_BAND_FIELDS = MappingProxyType({
    "RHOL": ("RHOL_VIS", "RHOL_NIR"),
    "RHOS": ("RHOS_VIS", "RHOS_NIR"),
    "TAUL": ("TAUL_VIS", "TAUL_NIR"),
    "TAUS": ("TAUS_VIS", "TAUS_NIR"),
})
_CATEGORY_IDENTITY_FIELDS = ("ISWATER", "ISBARREN", "ISICE", "ISCROP",
                             "EBLFOREST")
# module_sf_noahmpdrv.F lines 1464-1469: the urban branch fires for the
# built-up category and for every local-climate-zone category.
_URBAN_CATEGORY_KEYS = ("ISURBAN",) + tuple(f"LCZ_{n}" for n in range(1, 12))
# ALBSAT_TABLE(SOILCOLOR,:) / ALBDRY_TABLE(SOILCOLOR,:).
_RAD_SOILCOLOR_FIELDS = MappingProxyType({
    "ALBSAT": ("ALBSAT_VIS", "ALBSAT_NIR"),
    "ALBDRY": ("ALBDRY_VIS", "ALBDRY_NIR"),
})
_RAD_BAND_FIELDS = ("ALBICE", "ALBLAK", "OMEGAS", "EG")
_RAD_SCALAR_FIELDS = ("BETADS", "BETAIS")
_GLOBAL_SCALAR_FIELDS = (
    "CO2", "O2", "TIMEAN", "FSATMX", "Z0SNO", "SSI", "SNOW_RET_FAC",
    "SNOW_EMIS", "SWEMX", "TAU0", "GRAIN_GROWTH", "EXTRA_GROWTH",
    "DIRT_SOOT", "BATS_COSZ", "BATS_VIS_NEW", "BATS_NIR_NEW",
    "BATS_VIS_AGE", "BATS_NIR_AGE", "BATS_VIS_DIR", "BATS_NIR_DIR",
    "RSURF_SNOW", "RSURF_EXP",
)
# parameters%FIELD = FIELD_TABLE(SOILTYPE(1)) out of the tile-drain group.
_TILEDRAIN_SCALAR_FIELDS = MappingProxyType({
    "KLAT_FAC": "KLAT_FAC", "TDSMC_FAC": "TDSMC_FAC", "TD_DC": "TD_DC",
    "TD_DCOEF": "TD_DCOEF", "TD_RADI": "TD_RADI", "TD_SPAC": "TD_SPAC",
    "TD_DDRAIN": "TD_DDRAIN", "TD_ADEPTH": "TD_ADEPTH", "TD_D": "TD_D",
})
_TILEDRAIN_INTEGER_FIELDS = MappingProxyType({"TD_DEPTH": "TD_DEPTH"})
# read_mp_soil_parameters reads the first SOILPARM section positionally
# (module_sf_noahmplsm.F lines 11726-11731); these are its column names.
SOILPARM_SECTION = "STAS"
_SOIL_LAYER_FIELDS = MappingProxyType({
    "BEXP": "BB", "DKSAT": "SATDK", "DWSAT": "SATDW", "PSISAT": "SATPSI",
    "QUARTZ": "QTZ", "SMCDRY": "DRYSMC", "SMCMAX": "MAXSMC",
    "SMCREF": "REFSMC", "SMCWLT": "WLTSMC",
})
_SOIL_TOP_FIELDS = MappingProxyType({
    "F1": "F11_OR_HC", "BVIC": "BVIC", "AXAJ": "AXAJ", "BXAJ": "BXAJ",
    "XXAJ": "XXAJ", "BDVIC": "BDVIC", "GDVIC": "GDVIC", "BBVIC": "BBVIC",
})
_GENPARM_FIELDS = MappingProxyType({
    "REFDK": "REFDK_DATA", "REFKDT": "REFKDT_DATA", "CSOIL": "CSOIL_DATA",
    "ZBOT": "ZBOT_DATA", "CZIL": "CZIL_DATA",
})
# module_sf_noahmpdrv.F lines 1697-1701.
URBAN_SOIL_OVERRIDE = MappingProxyType({
    "SMCMAX": 0.45, "SMCREF": 0.42, "SMCWLT": 0.40, "SMCDRY": 0.40,
})
URBAN_CSOIL_OVERRIDE = 3.0e6
# module_sf_noahmpdrv.F line 1706: WRF leaves FRZX unset for soil type 14.
FRZX_UNDEFINED_SOILTYPE = 14


def _r4(value: float) -> float:
    """Round to WRF's default REAL, which is IEEE binary32 in this build."""

    return float(np.float32(value))


@dataclass(frozen=True)
class NoahmpTransferredParameters:
    """One column of ``TRANSFER_MP_PARAMETERS`` output.

    ``values`` is keyed by the WRF component name; every entry is a tuple so
    that scalars and arrays are addressed uniformly. ``field`` uses the same
    ``index`` convention as the pinned oracle fixture: 0 for a scalar, 1-based
    for an array element.
    """

    dataset_identifier: str
    vegtype: int
    soiltype: tuple[int, ...]
    slopetype: int
    soilcolor: int
    croptype: int
    values: Mapping[str, tuple[ParameterAtom, ...]]

    @property
    def urban_flag(self) -> bool:
        return bool(self.values["URBAN_FLAG"][0])

    def scalar(self, name: str) -> ParameterAtom:
        entry = self.values[name.upper()]
        if len(entry) != 1:
            raise ValueError(f"{name} is an array of {len(entry)} elements")
        return entry[0]

    def array(self, name: str) -> tuple[ParameterAtom, ...]:
        return self.values[name.upper()]

    def field(self, name: str, index: int) -> ParameterAtom:
        entry = self.values[name.upper()]
        if index == 0:
            if len(entry) != 1:
                raise ValueError(f"{name} is an array; index 0 is not a member")
            return entry[0]
        if not 1 <= index <= len(entry):
            raise IndexError(f"{name}({index}) outside 1..{len(entry)}")
        return entry[index - 1]


def transfer_mp_parameters(
    bundle: NoahmpParameterBundle,
    *,
    vegtype: int,
    soiltype: Sequence[int],
    slopetype: int,
    soilcolor: int,
    croptype: int = 0,
    dataset_identifier: str = DEFAULT_VEGETATION_DATASET,
) -> NoahmpTransferredParameters:
    """Reproduce WRF v4.6.1 ``TRANSFER_MP_PARAMETERS`` from the pinned tables.

    Faithful to ``phys/module_sf_noahmpdrv.F`` lines 1435-1710 for the
    non-crop, non-irrigation branch: ``croptype > 0`` and the irrigation block
    are refused rather than approximated, because the pinned option identity
    (``opt_crop = 0``, ``opt_irr = 0``) never assigns those components and the
    packaged MPTABLE offers no way to validate them.
    """

    soiltype = tuple(int(value) for value in soiltype)
    if not soiltype:
        raise ValueError("soiltype must name at least one layer")
    if croptype != 0:
        raise NotImplementedError(
            "TRANSFER_MP_PARAMETERS crop branch (CROPTYPE > 0) is not ported"
        )
    categories, veg = bundle.vegetation_groups(dataset_identifier)
    nveg = categories.scalar("NVEG")
    if not 1 <= vegtype <= nveg:
        raise ValueError(f"vegtype {vegtype} outside 1..{nveg}")
    rad = bundle.mptable["noahmp_rad_parameters"]
    glob = bundle.mptable["noahmp_global_parameters"]
    tile = bundle.mptable["noahmp_tiledrain_parameters"]
    soil = bundle.soil[SOILPARM_SECTION]
    slopes = bundle.general.slopes
    if not 1 <= slopetype <= len(slopes):
        raise ValueError(f"slopetype {slopetype} outside 1..{len(slopes)}")
    ncolor = len(rad.values["ALBSAT_VIS"])
    if not 1 <= soilcolor <= ncolor:
        raise ValueError(f"soilcolor {soilcolor} outside 1..{ncolor}")
    for category in soiltype:
        if not 1 <= category <= len(soil.rows):
            raise ValueError(
                f"soiltype {category} outside 1..{len(soil.rows)}"
            )

    values: dict[str, tuple[ParameterAtom, ...]] = {}

    def put(name: str, *entries: ParameterAtom) -> None:
        if name in values:
            raise ValueError(f"duplicate transferred field {name}")
        values[name] = tuple(entries)

    def veg_at(key: str) -> ParameterAtom:
        entry = veg.values[key]
        return entry[vegtype - 1] if len(entry) > 1 else entry[0]

    def tile_at(key: str) -> ParameterAtom:
        entry = tile.values[key]
        return entry[soiltype[0] - 1] if len(entry) > 1 else entry[0]

    # Category identity flags (lines 1457-1461).
    for name in _CATEGORY_IDENTITY_FIELDS:
        put(name, int(veg.scalar(name)))
    # Urban branch (lines 1463-1469).
    put("URBAN_FLAG", vegtype in {int(veg.scalar(k)) for k in _URBAN_CATEGORY_KEYS})

    # Vegetation block (lines 1475-1529).
    for name, key in _VEGETATION_SCALAR_FIELDS.items():
        put(name, _r4(veg_at(key)))
    for name, key in _VEGETATION_INTEGER_FIELDS.items():
        put(name, int(_r4(veg_at(key))))
    put("SAIM", *(_r4(veg_at(f"SAI_{month}")) for month in _MONTHS))
    put("LAIM", *(_r4(veg_at(f"LAI_{month}")) for month in _MONTHS))
    for name, keys in _VEGETATION_BAND_FIELDS.items():
        put(name, *(_r4(veg_at(key)) for key in keys))

    # Radiation block (lines 1535-1542).
    for name, keys in _RAD_SOILCOLOR_FIELDS.items():
        put(name, *(_r4(rad.values[key][soilcolor - 1]) for key in keys))
    for name in _RAD_BAND_FIELDS:
        put(name, *(_r4(entry) for entry in rad.values[name]))
    for name in _RAD_SCALAR_FIELDS:
        put(name, _r4(rad.scalar(name)))

    # Global block (lines 1607-1628).
    for name in _GLOBAL_SCALAR_FIELDS:
        put(name, _r4(glob.scalar(name)))

    # Tile-drainage block (lines 1672-1682).
    for name, key in _TILEDRAIN_SCALAR_FIELDS.items():
        put(name, _r4(tile_at(key)))
    for name, key in _TILEDRAIN_INTEGER_FIELDS.items():
        put(name, int(tile_at(key)))
    put("DRAIN_LAYER_OPT", int(tile.scalar("DRAIN_LAYER_OPT")))

    # Soil block (lines 1634-1654).
    for name, key in _SOIL_LAYER_FIELDS.items():
        column = soil.column(key)
        put(name, *(_r4(column[category - 1]) for category in soiltype))
    for name, key in _SOIL_TOP_FIELDS.items():
        put(name, _r4(soil.column(key)[soiltype[0] - 1]))

    # GENPARM block plus the derived quantities (lines 1687-1708).
    for name, key in _GENPARM_FIELDS.items():
        put(name, _r4(bundle.general.values[key]))
    frzk = _r4(bundle.general.values["FRZK_DATA"])
    put("KDT", _r4(_r4(values["REFKDT"][0] * values["DKSAT"][0])
                   / values["REFDK"][0]))
    put("SLOPE", _r4(slopes[slopetype - 1]))
    if values["URBAN_FLAG"][0]:
        for name, override in URBAN_SOIL_OVERRIDE.items():
            values[name] = (_r4(override),) * len(values[name])
        values["CSOIL"] = (_r4(URBAN_CSOIL_OVERRIDE),)
    if soiltype[0] != FRZX_UNDEFINED_SOILTYPE:
        frzfact = _r4(_r4(values["SMCMAX"][0] / values["SMCREF"][0])
                      * _r4(_r4(0.412) / _r4(0.468)))
        put("FRZX", _r4(frzk * frzfact))

    return NoahmpTransferredParameters(
        dataset_identifier=dataset_identifier,
        vegtype=int(vegtype),
        soiltype=soiltype,
        slopetype=int(slopetype),
        soilcolor=int(soilcolor),
        croptype=int(croptype),
        values=MappingProxyType(values),
    )


__all__ = [
    "DATASET_GROUPS",
    "DEFAULT_VEGETATION_DATASET",
    "FRZX_UNDEFINED_SOILTYPE",
    "MPTABLE_GROUPS",
    "NoahmpGeneralTable",
    "NoahmpNamelistGroup",
    "NoahmpParameterBundle",
    "NoahmpSoilRow",
    "NoahmpSoilTable",
    "NoahmpTransferredParameters",
    "SOILPARM_SECTION",
    "SOIL_PARAMETER_NAMES",
    "URBAN_CSOIL_OVERRIDE",
    "URBAN_SOIL_OVERRIDE",
    "load_noahmp_parameters",
    "parse_genparm",
    "parse_mptable",
    "parse_soilparm",
    "transfer_mp_parameters",
]
