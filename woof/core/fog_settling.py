"""WRF's gravitational settling of fog droplets: ``grav_settling`` 1 and 2.

Two WRF v4.7.1 routines (public domain), transcribed in
``kernels/fogdes.cu`` and replayed against captured native real-input calls
by ``tools/fog_kfeta_wrf471/replay.py``:

* ``sf_fogdes`` (phys/module_sf_fogdes.F), called by the surface driver
  after the surface layer and before the land surface
  (module_surface_driver.F:2494-2524): the fog deposition velocity VDFG,
  the step's deposition DFGDP (mm) and its accumulator FGDP, and RAINBL +=
  DFGDP, so the deposited water reaches the land surface as rain does.
  ``grav_settling = 1`` takes VDFG from Duynkerke's free-atmosphere fall
  speed; ``2`` is FogDES (Katata et al. 2008, 2011): a canopy deposition
  velocity from LAI and the surface-layer wind on vegetated categories, the
  settling of the mean droplet everywhere else.
* ``bl_fogdes`` (phys/module_bl_fogdes.F), called by the PBL driver right
  after the PBL scheme (phys/module_pbl_driver.F:2195-2205): the settling flux
  divergence of cloud water, added to RQCBLTEN, with the deposition flux
  qc*VDFG leaving the lowest level.

Both run on the surface/PBL cadence (``bldt``) with ``dtbl`` in seconds, from
the PBL slot and surface seams every PBL and surface-layer scheme shares in
:class:`woof.core.physics.PhysicsDriver`.  The land-use dependence
(FogDES's canopy classes) is the hash-pinned table
``woof/data/fogdes/fogdes_categories.v1.json``, not code.

The module keeps ``--ftz=false`` and calls the float32 math header's pow.
Strict numerics and ``GPUWM_WRF_EXACT=1`` also disable FMA contraction;
default fast numerics follows the shared compiler's contraction policy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

#: WRF ``&physics grav_settling`` values (Registry.EM_COMMON:2654 and the
#: fogsettling0/1/2 packages at :3237-3239), all ported.
GRAV_SETTLING_VALUES = (0, 1, 2)

#: The three per-column fields WRF's packages allocate (Registry
#: .EM_COMMON:1150-1152), carried in ``PhysicsDriver.fields`` and so
#: restart-carried with the rest of the surface inventory.
STATE_FIELDS = ("vdfg", "fgdp", "dfgdp")

#: Strict module options; default fast numerics permits FMA contraction.
MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.kernels:fogdes"
_SOURCES = ("glibc_flt32.cuh", "fogdes.cu")
_BLOCK = 128
_NULL = np.uintp(0)

TABLE_PATH = (Path(__file__).resolve().parents[1] / "data" / "fogdes"
              / "fogdes_categories.v1.json")
#: SHA-256 of :data:`TABLE_PATH`.  :func:`load_fogdes_table` refuses other
#: bytes: the oracle fixtures were graded against these classes, so a table
#: that moved is an unproven model.
TABLE_SHA256 = "ffa3a64cdbcc8c06699e44958d34aa229bf7fadebaefd44dc96d420f110b006f"


@dataclass(frozen=True)
class FogSettlingRefusal:
    """One configuration ``grav_settling`` 1 or 2 cannot run under.

    Rows, not code: :func:`refusal` walks them for the run door and the
    namelist importer, and ``tools/build_registry.py`` publishes each as a
    ``refused_when`` row on the named registry option.
    """

    component: str
    option: str
    selector: str
    value: int
    reason: str


#: Every configuration the option refuses, with the breakage each prevents.
REFUSED_WHEN: tuple[FogSettlingRefusal, ...] = ()
FORCED_SETTINGS = tuple({
    "selector": "mp_physics", "equals": mp, "setting": "grav_settling",
    "value": 0,
    "reason": "WRF 4.7.1 disables separate fog settling under mp=28/29 "
              "(share/module_check_a_mundo.F:2474-2487)",
} for mp in (28, 29))


def effective_grav_settling(value: int, *, mp_physics: int) -> int:
    for row in FORCED_SETTINGS:
        if int(mp_physics) == row["equals"]:
            return row["value"]
    return value


def refusal(*, grav_settling, mp_physics, bl_pbl_physics,
            sf_sfclay_physics) -> str | None:
    """Why ``grav_settling`` cannot run as set, or None.

    Shared by :func:`woof.config.validate_run_config` and the namelist
    importer, so both doors refuse the same configurations with the same
    words.
    """
    if (isinstance(grav_settling, bool)
            or not isinstance(grav_settling, (int, np.integer))
            or grav_settling not in GRAV_SETTLING_VALUES):
        return (f"grav_settling = {grav_settling!r} is not one of "
                f"{GRAV_SETTLING_VALUES} (0 off, 1 Duynkerke settling, "
                "2 FogDES deposition)")
    if grav_settling == 0:
        return None
    selectors = {"mp_physics": mp_physics, "bl_pbl_physics": bl_pbl_physics,
                 "sf_sfclay_physics": sf_sfclay_physics}
    for row in REFUSED_WHEN:
        value = selectors[row.selector]
        if value is not None and int(value) == row.value:
            return (f"grav_settling = {grav_settling} cannot run with "
                    f"{row.selector} = {row.value}: {row.reason}.  Set "
                    "grav_settling = 0")
    return None


# ---------------------------------------------------------------------------
# The FogDES land-use table.
# ---------------------------------------------------------------------------


class FogdesTableError(ValueError):
    """The FogDES category table is not the one the oracle graded."""


@dataclass(frozen=True)
class FogdesCategories:
    """One land-use dataset's FogDES classes, indexed by IVGTYP.

    ``vegetated[cat]`` is 1 for a canopy class and 0 for OTHERS;
    ``spcfct``/``vegh`` hold that class's float32 constants (0 for OTHERS).
    Index 0 is unused (categories are 1-based); a category past the end is
    OTHERS, as every category WRF does not list is.
    """

    dataset: str
    num_land_cat: tuple[int, ...]
    mminlu: tuple[str, ...]
    vegetated: np.ndarray
    spcfct: np.ndarray
    vegh: np.ndarray
    sha256: str

    @property
    def ncat(self) -> int:
        return int(self.vegetated.shape[0])


def _f32(entry: Mapping[str, str], where: str) -> np.float32:
    bits = int(entry["f32_bits"], 16)
    value = np.array([bits], dtype=np.uint32).view(np.float32)[0]
    decimal = np.float32(float(entry["decimal"]))
    if decimal.view(np.uint32) != np.uint32(bits):
        raise FogdesTableError(
            f"{where}: decimal {entry['decimal']} rounds to "
            f"{hex(int(decimal.view(np.uint32)))}, not the recorded "
            f"{entry['f32_bits']}")
    return value


@lru_cache(maxsize=None)
def load_fogdes_table(path: str | None = None, *,
                      verify_sha256: bool = True) -> Mapping[str, FogdesCategories]:
    """``{dataset: FogdesCategories}`` from the pinned table."""
    table = Path(path) if path is not None else TABLE_PATH
    if verify_sha256:
        from woof.core.thompson_contract import TableAsset, read_validated_table_asset
        raw = read_validated_table_asset(
            table.parent, TableAsset(table.name, 2904, TABLE_SHA256))
    else:
        raw = table.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_sha256 and digest != TABLE_SHA256:
        raise FogdesTableError(
            f"{table} has SHA-256 {digest}, not the graded table's "
            f"{TABLE_SHA256}: the FogDES oracle was graded against those "
            "classes, so another table is an unproven model")
    doc = json.loads(raw.decode("utf-8"))
    classes = {
        name: (_f32(row["spcfct"], f"{name}.spcfct"),
               _f32(row["vegh"], f"{name}.vegh"))
        for name, row in doc["classes"].items()}
    out = {}
    for dataset, row in doc["datasets"].items():
        listed = [cat for cats in row["classes"].values() for cat in cats]
        if len(listed) != len(set(listed)) or min(listed) < 1:
            raise FogdesTableError(
                f"{dataset}: a category is listed twice or below 1")
        ncat = max(listed) + 1
        vegetated = np.zeros(ncat, dtype=np.int8)
        spcfct = np.zeros(ncat, dtype=np.float32)
        vegh = np.zeros(ncat, dtype=np.float32)
        for name, cats in row["classes"].items():
            if name not in classes:
                raise FogdesTableError(f"{dataset}: unknown class {name}")
            for cat in cats:
                vegetated[cat] = 1
                spcfct[cat], vegh[cat] = classes[name]
        for array in (vegetated, spcfct, vegh):
            array.setflags(write=False)
        out[dataset] = FogdesCategories(
            dataset=dataset, num_land_cat=tuple(row["num_land_cat"]),
            mminlu=tuple(row["mminlu"]), vegetated=vegetated, spcfct=spcfct,
            vegh=vegh, sha256=digest)
    return MappingProxyType(out)


def categories_for_num_land_cat(num_land_cat: int) -> FogdesCategories:
    """Legacy 4.6.1 fixture lookup; production uses 4.7.1 MMINLU below."""
    for categories in load_fogdes_table().values():
        if int(num_land_cat) in categories.num_land_cat:
            return categories
    raise FogdesTableError(_unmapped(f"num_land_cat = {num_land_cat}"))


def categories_for_landuse(landuse_dataset: str | None) -> FogdesCategories:
    """The classes for a domain's land-use dataset (its MMINLU)."""
    for categories in load_fogdes_table().values():
        if landuse_dataset == {"MODIS": "MODIFIED_IGBP_MODIS_NOAH", "USGS": "USGS"}[categories.dataset]:
            return categories
    raise FogdesTableError(_unmapped(f"land-use dataset {landuse_dataset!r}"))


def _unmapped(what: str) -> str:
    known = {name: (c.mminlu, c.num_land_cat)
             for name, c in load_fogdes_table().items()}
    return (f"grav_settling = 1/2 checks its land-use table by "
            f"land-use category, and {what} is not one of the datasets WRF "
            f"classifies {known}; WRF itself stops here "
            "(module_sf_fogdes.F:114-121).  Use a USGS or MODIS land-use "
            "dataset (WRF 4.7.1 checks MMINLU for both options)")


# ---------------------------------------------------------------------------
# The device module.
# ---------------------------------------------------------------------------


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import module_source as source_for
    return source_for("fogdes")


def compile_module(source: str):
    """Compile a production or mutation unit through the shared loader."""
    from woof.core.kernels import compile_noftz_module
    key = MODULE_KEY
    if source != module_source():
        key += ":mutation:" + hashlib.sha256(source.encode()).hexdigest()
    return compile_noftz_module("fogdes", source, key)


@lru_cache(maxsize=None)
def _module(device: int):
    import cupy as cp

    with cp.cuda.Device(device):
        from woof.core.kernels import load_module
        return load_module("fogdes")


def _kernel(name: str, module=None):
    import cupy as cp

    if module is None:
        module = _module(cp.cuda.Device().id)
    return module.get_function(name)


@lru_cache(maxsize=None)
def _device_table(dataset: str, sha256: str, device: int):
    import cupy as cp

    categories = load_fogdes_table()[dataset]
    with cp.cuda.Device(device):
        return (cp.asarray(categories.vegetated), cp.asarray(categories.spcfct),
                cp.asarray(categories.vegh))


def _flat(xp, array, name: str, *, writable: bool = False, dtype=None):
    dtype = np.float32 if dtype is None else dtype
    if writable:
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise TypeError(
                f"fog settling writes {name} in place, which must be a "
                f"C-contiguous {np.dtype(dtype).name} array: a converted "
                "copy would take the write and the field would never see it")
        return array
    return xp.ascontiguousarray(xp.asarray(array, dtype=dtype))


def surface_deposition(*, grav_settling: int, qc1, rho1, dz1, ivgtyp, lai,
                       wspd, vdfg, fgdp, dfgdp, dtbl,
                       categories: FogdesCategories | None, rainbl=None,
                       module=None) -> None:
    """WRF's sf_fogdes on every column, in place.

    ``qc1``/``rho1``/``dz1`` are the lowest mass level's cloud water, density
    and dz8w; every array is per column (any 2-D or 1-D shape, the same for
    all).  VDFG, DFGDP and FGDP are written in place; with ``rainbl`` the
    driver's RAINBL += DFGDP follows (module_surface_driver.F:2521-2522).
    ``categories`` is required for ``grav_settling = 2`` only.
    """
    import cupy as cp

    gs = int(grav_settling)
    if gs not in (1, 2):
        raise ValueError(
            f"surface_deposition runs for grav_settling 1 and 2, not {gs}: "
            "WRF's surface driver zeroes VDFG instead (module_surface_driver"
            ".F:2494-2500)")
    if gs == 2 and categories is None:
        raise ValueError("grav_settling = 2 needs the FogDES land-use classes")
    ncol = int(vdfg.size)
    args_in = [_flat(cp, a, n) for a, n in
               ((qc1, "qc1"), (rho1, "rho1"), (dz1, "dz1"))]
    ivg = _flat(cp, ivgtyp, "ivgtyp", dtype=np.int32)
    lai_ = _flat(cp, lai, "lai")
    wspd_ = _flat(cp, wspd, "wspd")
    outs = [_flat(cp, a, n, writable=True) for a, n in
            ((vdfg, "vdfg"), (fgdp, "fgdp"), (dfgdp, "dfgdp"))]
    rain = _NULL if rainbl is None else _flat(cp, rainbl, "rainbl",
                                              writable=True)
    for array, name in zip(args_in + [ivg, lai_, wspd_] + outs,
                           ("qc1", "rho1", "dz1", "ivgtyp", "lai", "wspd",
                            "vdfg", "fgdp", "dfgdp")):
        if int(array.size) != ncol:
            raise ValueError(f"{name} has {array.size} columns, VDFG {ncol}")
    if categories is None:
        vegetated = cp.zeros(1, dtype=cp.int8)
        spcfct = vegh = cp.zeros(1, dtype=cp.float32)
        ncat = 1
    else:
        vegetated, spcfct, vegh = _device_table(
            categories.dataset, categories.sha256, cp.cuda.Device().id)
        ncat = categories.ncat
    if ncol == 0:
        return
    _kernel("fogdes_surface", module)(
        ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (*args_in, ivg, lai_, wspd_, vegetated, spcfct, vegh, np.int32(ncat),
         *outs, rain, np.float32(dtbl), np.int32(gs), np.int64(ncol)))


def settling_tendency(*, grav_settling: int, qc, dz, vdfg, dqc,
                      module=None) -> None:
    """WRF's bl_fogdes: add the settling flux divergence to ``dqc`` in place.

    ``qc``, ``dz`` and ``dqc`` are (nz, ...) on the mass levels, ``kts`` at
    index 0 and ``kte`` last; ``vdfg`` is this step's deposition velocity.
    """
    import cupy as cp

    if int(grav_settling) not in (1, 2):
        raise ValueError(
            f"settling_tendency runs for grav_settling 1 and 2, not "
            f"{grav_settling}: WRF's PBL driver skips bl_fogdes at 0 "
            "(phys/module_pbl_driver.F:2195)")
    nz = int(dqc.shape[0])
    if nz < 2:
        raise ValueError(f"bl_fogdes needs at least two levels, not {nz}")
    ncol = int(dqc.size) // nz
    qc_ = _flat(cp, qc, "qc")
    dz_ = _flat(cp, dz, "dz")
    vdfg_ = _flat(cp, vdfg, "vdfg")
    out = _flat(cp, dqc, "dqc", writable=True)
    if qc_.size != out.size or dz_.size != out.size or vdfg_.size != ncol:
        raise ValueError("qc, dz, dqc and vdfg disagree on the column count")
    if ncol == 0:
        return
    _kernel("fogdes_column", module)(
        ((ncol + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (qc_, dz_, vdfg_, out, np.int32(nz), np.int64(ncol)))


# ---------------------------------------------------------------------------
# The PhysicsDriver carrier.
# ---------------------------------------------------------------------------


class FogSettling:
    """The ``grav_settling`` carrier a :class:`PhysicsDriver` holds.

    Holds only the option and the land-use classes, both rebuilt from the
    resumed configuration; the state it writes (VDFG, FGDP, DFGDP) lives in
    the driver's ``fields`` and rides the checkpoint with them.
    """

    def __init__(self, grav_settling: int,
                 categories: FogdesCategories | None) -> None:
        if int(grav_settling) not in (1, 2):
            raise ValueError(
                f"FogSettling is the carrier for grav_settling 1 and 2, not "
                f"{grav_settling}")
        self.grav_settling = int(grav_settling)
        self.categories = categories

    def setup_identity(self) -> dict:
        """Bind source, arithmetic and active coefficient bytes to restarts."""
        return {
            "algorithm": "wrf-4.7.1-fogdes-v1",
            "source_sha256": hashlib.sha256(module_source().encode()).hexdigest(),
            "compile_options": list(MODULE_OPTIONS),
            "grav_settling": self.grav_settling,
            "landuse": None if self.categories is None else self.categories.dataset,
            "table_sha256": None if self.categories is None else self.categories.sha256,
        }

    def surface(self, fields, atmosphere, *, dtbl: float, rainbl=None) -> None:
        """sf_fogdes after the surface layer, before the land surface."""
        fields["dfgdp"][...] = 0.0
        surface_deposition(
            grav_settling=self.grav_settling,
            qc1=atmosphere["qc"][0], rho1=atmosphere["rho"][0],
            dz1=atmosphere["dz"][0], ivgtyp=fields["ivgtyp"],
            lai=fields["lai"], wspd=fields["wspd"], vdfg=fields["vdfg"],
            fgdp=fields["fgdp"], dfgdp=fields["dfgdp"], dtbl=dtbl,
            categories=self.categories, rainbl=rainbl)

    def pbl_rates(self, rates: Mapping, atmosphere, vdfg) -> dict:
        """The PBL slot's rates with bl_fogdes added to ``dqc``.

        A new ``dqc`` array, never the scheme's own: a scheme may hand a
        buffer it keeps, and adding in place would accumulate across calls.
        """
        import cupy as cp

        source = rates.get("dqc")
        dqc = (cp.zeros_like(atmosphere["qc"]) if source is None
               else cp.array(source, dtype=cp.float32, order="C", copy=True))
        settling_tendency(grav_settling=self.grav_settling,
                          qc=atmosphere["qc"], dz=atmosphere["dz"],
                          vdfg=vdfg, dqc=dqc)
        return {**rates, "dqc": dqc}


def build_fog_settling(cfg, landuse_dataset: str | None) -> FogSettling | None:
    """The carrier for ``cfg.grav_settling``, or None at 0.

    Refused by name where the option cannot run: the configuration rows
    (:func:`refusal`), and FogDES on a land-use dataset WRF does not
    classify.
    """
    gs = effective_grav_settling(
        int(getattr(cfg, "grav_settling", 0) or 0), mp_physics=cfg.mp_physics)
    why = refusal(grav_settling=gs, mp_physics=cfg.mp_physics,
                  bl_pbl_physics=cfg.bl_pbl_physics,
                  sf_sfclay_physics=cfg.sf_sfclay_physics)
    if why is not None:
        raise ValueError(why)
    if gs == 0:
        return None
    # surface_driver returns at sf_sfclay_physics=0 before sf_fogdes and
    # its MMINLU check (WRF 4.7.1 module_surface_driver.F:1489-1490).
    # bl_fogdes still runs and never reads the land-use categories.
    categories = (categories_for_landuse(landuse_dataset)
                  if cfg.sf_sfclay_physics else None)
    return FogSettling(gs, categories)


__all__ = [
    "FogSettling", "FogSettlingRefusal", "FogdesCategories", "FogdesTableError",
    "GRAV_SETTLING_VALUES", "MODULE_KEY", "MODULE_OPTIONS", "REFUSED_WHEN",
    "STATE_FIELDS", "TABLE_PATH", "TABLE_SHA256", "build_fog_settling",
    "categories_for_landuse", "categories_for_num_land_cat", "compile_module",
    "load_fogdes_table", "module_source", "refusal", "settling_tendency",
    "surface_deposition",
]
